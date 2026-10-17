"""Full target crop-selection ablation; frozen detector, no score-based gate.

Reuse the source pilot's selectors, two-crop geometry, held prompts and 0.5 fusion.
Official category-pooled metrics are computed independently for each random seed;
random_mean is the arithmetic mean of metrics, NOT a map/prediction ensemble.
Optional previous evaluation caches provide a per-image equivalence guard.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from audit_reinspection_risk import safe_child
from calibrate_reinspection import CONFIG_KEYS
from source_selection_control import inspect_image, prepare as prepare_source, average


def prepare(args):
    source_path, source, baseline, _, output, source_reference = prepare_source(args)
    if args.testing_data in source['metadata']['sources']:
        raise ValueError('Target must differ from auxiliary training sources')
    previous = None
    if args.reference_eval:
        path = Path(args.reference_eval).resolve()
        if output in path.parents or path.parent in output.parents:
            raise ValueError('Output must be separate from reference target experiment')
        report = json.loads(path.read_text(encoding='utf8'))
        if (report.get('status') != 'complete' or report.get('testing_data') != args.testing_data
                or report.get('unchanged_detector_tensors', 0) <= 0):
            raise ValueError('Require a completed frozen-detector reference for the same target')
        for key in CONFIG_KEYS:
            if report['fixed_rules']['config'].get(key) != source['metadata'].get(key):
                raise ValueError('Source/reference target configuration mismatch: ' + key)
        records = {str(Path(r['image']).resolve()): r for r in report['per_image']}
        classes = list(report['metrics'])
        if not records or len(records) != len(report['per_image']) or set(classes) != {r['class'] for r in records.values()}:
            raise ValueError('Invalid reference image/class identities')
        arrays = {}
        side = source['metadata']['image_size']
        for i, category in enumerate(classes):
            rows = [r for r in records.values() if r['class'] == category]
            if sorted(r['cache_position'] for r in rows) != list(range(len(rows))):
                raise ValueError('Invalid reference cache positions')
            arrays[category] = {key: np.load(safe_child(path.parent, 'cache/class_%02d_%s.npy' % (i, key)),
                                            mmap_mode='r', allow_pickle=False)
                                for key in ('baseline', 'fused', 'mask')}
            if (any(a.shape != (len(rows), side, side) for a in arrays[category].values())
                    or arrays[category]['mask'].dtype != np.bool_):
                raise ValueError('Invalid reference cache shape/dtype')
        previous = dict(path=path, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        report=report, records=records, arrays=arrays)
    return source_path, source, baseline, output, source_reference, previous


def check_reference(prediction, image, category, previous, guard):
    """No target label input; old predictions are checked, never fitted to."""
    if previous is None:
        return
    row = previous['records'][str(Path(image).resolve())]
    if row['class'] != category:
        raise ValueError('Reference category mismatch')
    cache = previous['arrays'][category]
    for current, prior in (('baseline', 'baseline'), ('adaptive', 'fused')):
        a, b = prediction['maps'][current], cache[prior][row['cache_position']]
        if not np.isfinite(b).all() or not np.allclose(a, b, atol=1e-6, rtol=1e-5):
            raise RuntimeError('Reference %s map equivalence failed: %s' % (current, image))
        key = current + '_max_map_delta'
        guard[key] = max(guard[key], float(np.abs(a - b).max()))
    score_delta = abs(prediction['image_score'] - row['image_score'])
    if not math.isclose(prediction['image_score'], row['image_score'], abs_tol=1e-6, rel_tol=1e-5):
        raise RuntimeError('Reference global image score equivalence failed: ' + image)
    guard['max_image_score_delta'] = max(guard['max_image_score_delta'], score_delta)
    guard['checked_images'] += 1


def pooled_metrics(masks, maps, labels, scores, calculate_px, calculate_im):
    """Pool pixels WITH normal images, then use the original official helpers."""
    ap_im, auroc_im, f1_im = calculate_im(labels, scores)
    result = {}
    for mode, values in maps.items():
        auroc_px, f1_px, ap_px = calculate_px(masks, values)
        record = dict(auroc_px=float(auroc_px), f1_px=float(f1_px), ap_px=float(ap_px),
                      auroc_im=float(auroc_im), f1_im=float(f1_im), ap_im=float(ap_im))
        if not all(math.isfinite(v) for v in record.values()):
            raise RuntimeError('Non-finite pooled metric')
        result[mode] = record
    randoms = [v for k, v in result.items() if k.startswith('random_')]
    if not randoms:
        raise ValueError('At least one random-seed control is required')
    result['random_mean'] = {k: average([r[k] for r in randoms]) for k in result['baseline']}
    return result


def geometry_summary(rows, modes):
    result = {}
    for mode in modes:
        if mode == 'baseline':
            continue
        normal = [r for r in rows if r['anomaly'] == 0]
        result[mode] = dict(images=len(rows), valid_images=sum(r['selections'][mode]['geometry_valid'] for r in rows),
            valid_normal_images=sum(r['selections'][mode]['geometry_valid'] for r in normal),
            valid_anomalous_images=sum(r['selections'][mode]['geometry_valid'] for r in rows if r['anomaly'] == 1),
            mean_update_fraction=average([r['selections'][mode]['update_fraction'] for r in rows]),
            mean_defect_coverage=average([r['defect_coverage'][mode] for r in rows if r['defect_coverage'][mode] is not None]))
    return result


def main(args):
    source_path, source, baseline, output, source_reference, previous = prepare(args)
    from dataset import get_data
    from method import AdaCLIP_Trainer
    from tools import Logger, setup_seed, write2csv, calculate_average_metric
    from tools.metrics import calculate_px_metrics, calculate_im_metrics
    from tools.gate_equivalence import _rng_state, _restore_rng
    from tools.spatial_channel_pilot import import_baseline, frozen_guard

    meta = source['metadata']
    modes = ['baseline', 'adaptive', 'fixed_center'] + ['random_%d' % s for s in args.random_seeds]
    protocol = dict(source_report=str(source_path), source_report_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        source_config={k: meta[k] for k in CONFIG_KEYS}, baseline_checkpoint=str(baseline),
        testing_data=args.testing_data, random_seeds=args.random_seeds, crops_per_variant=2,
        fusion_weight=.5, no_training=True, no_score_gate=True, no_target_threshold_tuning=True,
        image_score_policy='same original global AdaCLIP/HSF score for every variant',
        rng_policy='one initial seed; preserve baseline trajectory, matching evaluate_reinspection.py',
        random_identity='SHA256(seed:target_name:absolute_image_path), first 32 bits -> numpy RandomState',
        geometry_safety='same source-pilot own-anchor/core/ring test for every selector; no score threshold',
        metric_protocol='official category-pooled pixels including normal images, then unweighted category mean; sigma4 once',
        random_mean='mean of independently computed seed metrics, NOT an ensemble and not best-seed selection',
        target_results_are_exploratory=True,
        per_image_model_calls_excluding_first_equivalence_check=1 + 2 * (len(modes) - 1),
        reference_eval=str(previous['path']) if previous else None,
        reference_eval_sha256=previous['sha256'] if previous else None)
    setup_seed(meta['seed'])
    config = json.loads((Path(__file__).parent / 'model_configs' / (meta['model'] + '.json')).read_text())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'cache').mkdir()
    (output / 'csvs').mkdir()
    logger = Logger(str(output / 'selection_control_eval.txt'))
    (output / 'fixed_selection_protocol.json').write_text(json.dumps(protocol, indent=2, allow_nan=False), encoding='utf8')
    logger.info('FIXED SELECTION PROTOCOL: ' + json.dumps(protocol, allow_nan=False))
    trainer = AdaCLIP_Trainer(backbone=meta['model'],
        feat_list=[config['vision_cfg']['layers'] // 4 * i for i in (1, 2, 3, 4)],
        input_dim=config['vision_cfg']['width'], output_dim=config['embed_dim'], learning_rate=0.,
        device=device, image_size=meta['image_size'], prompting_depth=meta['prompting_depth'],
        prompting_length=meta['prompting_length'], prompting_branch=meta['prompting_branch'], prompting_type=meta['prompting_type'],
        use_hsf=meta['use_hsf'], k_clusters=meta['k_clusters'], fusion_mode='add').to(device)
    before = import_baseline(trainer, str(baseline))
    for p in trainer.parameters():
        p.requires_grad_(False)
        p.grad = None
    trainer.eval()
    versions = {k: v._version for k, v in trainer.clip_model.state_dict().items()}
    classes, dataset, _ = get_data(dataset_type_list=args.testing_data, transform=trainer.preprocess,
                                  target_transform=trainer.transform, training=False)
    if getattr(dataset, 'training', True):
        raise ValueError('Target augmentation must be disabled')
    counts = {name: sum(r['cls_name'] == name for r in dataset.data_all) for name in classes}
    if (not counts or len(classes) != len(counts) or any(n == 0 for n in counts.values())
            or sum(counts.values()) != len(dataset)):
        raise ValueError('Invalid target class counts')
    identities = [str((Path(dataset.root) / r['img_path']).resolve()) for r in dataset.data_all]
    source_ids = {str(Path(r['image']).resolve()) for split in ('train', 'validation')
                  for r in source_reference[split]['per_image']}
    if len(set(identities)) != len(identities) or set(identities) & source_ids:
        raise ValueError('Duplicate target identities or overlap with source pilot')
    if previous and (set(identities) != set(previous['records']) or
                     counts != {c: len(a['baseline']) for c, a in previous['arrays'].items()}):
        raise ValueError('Reference target image identities/class counts differ')
    side = meta['image_size']
    disk_bytes = len(dataset) * side * side * (4 * len(modes) + 1)
    logger.info('Target images=%d; map caches estimated %.2f GiB; two crops PER variant, random trials are separate.' %
                (len(dataset), disk_bytes / 1024 ** 3))
    cache, class_rows = {}, {name: [] for name in classes}
    for i, name in enumerate(classes):
        cache[name] = {mode: np.lib.format.open_memmap(
            output / 'cache' / ('class_%02d_%s.npy' % (i, mode)), mode='w+',
            dtype=np.bool_ if mode == 'mask' else np.float32, shape=(counts[name], side, side))
            for mode in modes + ['mask']}
    guard = dict(enabled=previous is not None, checked_images=0, baseline_max_map_delta=0., adaptive_max_map_delta=0.,
                 max_image_score_delta=0., atol=1e-6, rtol=1e-5)
    report = dict(status='running', testing_data=args.testing_data, protocol=protocol, cache_estimated_bytes=disk_bytes,
                  per_image=[], metrics={}, reference_equivalence=guard)
    report_path = output / 'selection_control_eval.json'
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    start = time.perf_counter()
    for i, items in enumerate(loader):
        image_path, category = items['img_path'][0], items['cls_name'][0]
        with Image.open(image_path) as image:
            original = image.convert('RGB')
        prediction = inspect_image(trainer, items['img'], items['cls_name'], original, meta,
            args.testing_data + ':' + str(Path(image_path).resolve()), args.random_seeds,
            (_rng_state, _restore_rng), check=i == 0)
        if i == 0:
            report['initial_equivalence'] = prediction['initial_equivalence']
        check_reference(prediction, image_path, category, previous, guard)
        # GT and anomaly labels are consumed ONLY after all selections/inference.
        mask = items['img_mask'][0, 0].numpy() > .5
        anomaly = int(items['anomaly'][0])
        if anomaly not in (0, 1) or (anomaly == 0 and mask.any()) or mask.shape != (side, side):
            raise ValueError('Invalid target GT')
        if previous:
            old = previous['records'][str(Path(image_path).resolve())]
            if old['anomaly'] != anomaly or not np.array_equal(mask, previous['arrays'][category]['mask'][old['cache_position']]):
                raise ValueError('Reference target GT differs')
        position = len(class_rows[category])
        for mode, values in prediction['maps'].items():
            if values.shape != (side, side):
                raise ValueError('Invalid predicted map shape')
            cache[category][mode][position] = values
        cache[category]['mask'][position] = mask
        normal_response = {mode: dict(mean=float(values.mean()), q99=float(np.quantile(values, .99)))
                           for mode, values in prediction['maps'].items()} if anomaly == 0 else None
        row = dict(image=image_path, **{'class': category}, anomaly=anomaly, cache_position=position,
            image_score=prediction['image_score'], baseline_seconds=prediction['baseline_seconds'],
            selections=prediction['selections'], normal_response=normal_response,
            defect_coverage={mode: float(mask[support].sum() / mask.sum()) if mask.any() else None
                             for mode, support in prediction['supports'].items()})
        class_rows[category].append(row)
        report['per_image'].append(row)
        if (i + 1) % 32 == 0 or i + 1 == len(dataset):
            logger.info('Selection control progress=%d/%d elapsed=%.1fs' % (i + 1, len(dataset), time.perf_counter() - start))
            report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    if len(report['per_image']) != len(identities) or {str(Path(r['image']).resolve()) for r in report['per_image']} != set(identities):
        raise RuntimeError('Target evaluation identity mismatch')
    report['inference_loop_seconds'] = time.perf_counter() - start
    report['unchanged_detector_tensors'] = frozen_guard(trainer, before, versions)
    guard['passed'] = guard['checked_images'] == len(dataset) if previous else None
    all_modes = modes + ['random_mean']
    for name in classes:
        rows = class_rows[name]
        if len(rows) != counts[name]:
            raise RuntimeError('Missing target samples: ' + name)
        for values in cache[name].values():
            values.flush()
        logger.info('Computing official category-pooled metrics: ' + name)
        report['metrics'][name] = pooled_metrics(cache[name]['mask'], {mode: cache[name][mode] for mode in modes},
            np.asarray([r['anomaly'] for r in rows]), np.asarray([r['image_score'] for r in rows]),
            calculate_px_metrics, calculate_im_metrics)
        for mode, values in report['metrics'][name].items():
            write2csv(values, list(classes) + ['Average'], name,
                      str(output / 'csvs' / ('%s_selection_%s.csv' % (args.testing_data, mode))))
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    report['average'] = {mode: calculate_average_metric({name: report['metrics'][name][mode] for name in classes})
                         for mode in all_modes}
    report['random_seed_metric_std'] = {k: float(np.std([report['average']['random_%d' % s][k] for s in args.random_seeds]))
                                        for k in report['average']['baseline']}
    report['selection_summary'] = geometry_summary(report['per_image'], modes)
    report['selection_summary_by_class'] = {name: geometry_summary(rows, modes) for name, rows in class_rows.items()}
    normals = [r for r in report['per_image'] if r['anomaly'] == 0]
    report['normal_response_summary'] = {mode: {key: dict(
        image_average=average([r['normal_response'][mode][key] for r in normals]),
        mean_delta_vs_baseline=average([r['normal_response'][mode][key] - r['normal_response']['baseline'][key] for r in normals]))
        for key in ('mean', 'q99')} for mode in modes}
    report['normal_response_note'] = 'Unthresholded response diagnostics on normal images, NOT false-positive rates.'
    base_seconds = sum(r['baseline_seconds'] for r in report['per_image'])
    report['timing'] = dict(baseline_forward_seconds=base_seconds,
        variants={mode: dict(two_crop_forward_seconds=sum(sum(r['selections'][mode]['crop_seconds']) for r in report['per_image']))
                  for mode in modes if mode != 'baseline'},
        all_trials_computed_in_one_run=True, no_early_exit_or_compute_savings_claim=True)
    for values in report['timing']['variants'].values():
        values['model_forward_ratio_vs_baseline'] = (base_seconds + values['two_crop_forward_seconds']) / base_seconds if base_seconds else None
    for mode, values in report['average'].items():
        write2csv(values, list(classes) + ['Average'], 'Average',
                  str(output / 'csvs' / ('%s_selection_%s.csv' % (args.testing_data, mode))))
        logger.info('%s Average I-Auroc:%.2f I-F1:%.2f I-AP:%.2f P-Auroc:%.2f P-F1:%.2f P-AP:%.2f' %
                    (mode, values['auroc_im'], values['f1_im'], values['ap_im'], values['auroc_px'], values['f1_px'], values['ap_px']))
    report['status'] = 'complete'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    logger.info('Selection control target evaluation DONE; frozen detector PASS; reference equivalence=' + str(guard['passed']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--source_report', required=True)
    parser.add_argument('--reference_eval', help='Previous reinspection_eval.json AND its cache directory must remain available')
    parser.add_argument('--testing_data', choices=('mpdd', 'visa', 'btad'), default='mpdd')
    parser.add_argument('--random_seeds', nargs='+', type=int, default=[111, 222, 333])
    parser.add_argument('--save_path', required=True)
    main(parser.parse_args())
