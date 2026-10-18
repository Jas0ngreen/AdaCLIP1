"""Independent baseline/single-A/dual latency with crop-count cache replay.

No training, thresholds, metric selection, or new prediction caches. Timing is
batch-one file-to-CPU-output inference (warm filesystem cache), not model load,
GT loading, reference checks, report I/O, or cold-start application latency.
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
from source_selection_control import inspect_standalone, prepare as prepare_source


MODES = ('baseline', 'single_a', 'adaptive')
CALLS = dict(baseline=1, single_a=2, adaptive=3)


def prepare(args):
    if args.repeats < 1 or args.warmup < 1:
        raise ValueError('repeats and warmup must be positive')
    source_path, source, baseline, _, output, source_reference = prepare_source(args)
    path = Path(args.reference_eval).resolve()
    if output in path.parents or path.parent in output.parents:
        raise ValueError('Output must be separate from crop-count reference experiment')
    reference = json.loads(path.read_text(encoding='utf8'))
    protocol = reference.get('protocol', {})
    if (reference.get('status') != 'complete' or reference.get('testing_data') != args.testing_data
            or reference.get('unchanged_detector_tensors', 0) <= 0
            or protocol.get('experiment') != 'matched_support_crop_count_ablation'
            or protocol.get('background_alignment_used') is not False
            or protocol.get('full_single_crop_support_evaluated') is not False
            or protocol.get('fusion_weight') != .5
            or protocol.get('crops_per_variant') != dict(baseline=0, single_a=1, single_b=1, adaptive=2)):
        raise ValueError('Require completed unmodified crop_count_eval.json for the same target')
    if args.testing_data in source['metadata']['sources']:
        raise ValueError('Target must differ from training sources')
    for key in CONFIG_KEYS:
        if protocol['source_config'].get(key) != source['metadata'].get(key):
            raise ValueError('Crop-count/source configuration mismatch: ' + key)
    rows = reference['per_image']
    identities = [str(Path(r['image']).resolve()) for r in rows]
    classes = list(reference['metrics'])
    if not rows or len(set(identities)) != len(rows) or set(classes) != {r['class'] for r in rows}:
        raise ValueError('Invalid reference identities/classes')
    side, arrays = source['metadata']['image_size'], {}
    for i, category in enumerate(classes):
        part = [r for r in rows if r['class'] == category]
        if sorted(r['cache_position'] for r in part) != list(range(len(part))):
            raise ValueError('Invalid reference cache positions')
        arrays[category] = {mode: np.load(safe_child(path.parent, 'cache/class_%02d_%s.npy' % (i, mode)),
                                        mmap_mode='r', allow_pickle=False) for mode in MODES + ('mask',)}
        for mode, values in arrays[category].items():
            if (values.shape != (len(part), side, side)
                    or values.dtype != (np.bool_ if mode == 'mask' else np.float32)):
                raise ValueError('Invalid crop-count cache shape/dtype')
    return source_path, source, baseline, output, source_reference, path, reference, arrays


def predict_path(trainer, path, category, config, mode, rng_tools):
    """No GT, label, reference prediction, or other-mode output enters inference."""
    if str(path).endswith('.tif'):
        # Legacy global TIFF input uses OpenCV, but its native crop source
        # uses PIL. Preserve both paths rather than silently changing crops.
        import cv2
        global_image = Image.fromarray(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
        with Image.open(path) as image:
            original = image.convert('RGB')
    else:
        with Image.open(path) as image:
            original = image.convert('RGB')
        global_image = original
    # default_collate stacks the original evaluator's global input contiguously.
    # Preserve that layout (also for custom preprocessors) to avoid changing
    # floating-point reduction order in the global conditioning forward.
    image = trainer.preprocess(global_image).unsqueeze(0).contiguous()
    return inspect_standalone(trainer, image, [category], original, config, mode, rng_tools)


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def timed_prediction(trainer, path, category, config, mode, rng_tools):
    synchronize(trainer.device)
    start = time.perf_counter()
    prediction = predict_path(trainer, path, category, config, mode, rng_tools)
    synchronize(trainer.device)
    return prediction, time.perf_counter() - start


def check_prediction(prediction, row, arrays, mode, guard):
    expected = arrays[row['class']][mode][row['cache_position']]
    actual = prediction['map']
    if (not np.isfinite(expected).all() or actual.shape != expected.shape
            or not np.allclose(actual, expected, atol=1e-6, rtol=1e-5)):
        raise RuntimeError('Standalone %s map equivalence failed: %s' % (mode, row['image']))
    delta = abs(prediction['image_score'] - row['image_score'])
    if not math.isclose(prediction['image_score'], row['image_score'], abs_tol=1e-6, rel_tol=1e-5):
        raise RuntimeError('Standalone image score equivalence failed: ' + row['image'])
    if prediction['model_calls'] != CALLS[mode]:
        raise RuntimeError('Unexpected standalone model-call budget')
    if mode != 'baseline':
        prior = row['selections'][mode]
        if (int(prediction['support'].sum()) != prior['update_pixels']
                or prediction['geometry_valid'] != prior['geometry_valid']
                or prediction['selection']['boxes_native'] != prior['boxes_native']):
            raise RuntimeError('Standalone support/selection differs from reference')
    guard['checked_images'] += 1
    guard['max_map_delta'] = max(guard['max_map_delta'], float(np.abs(actual - expected).max()))
    guard['max_image_score_delta'] = max(guard['max_image_score_delta'], delta)


def new_guard():
    return dict(checked_images=0, max_map_delta=0., max_image_score_delta=0., atol=1e-6, rtol=1e-5)


def latency_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or np.min(values) <= 0:
        raise ValueError('Invalid latency observations')
    return dict(images=int(values.size), total_seconds=float(values.sum()),
                mean_ms=float(values.mean() * 1000), median_ms=float(np.median(values) * 1000),
                p95_ms=float(np.quantile(values, .95) * 1000), images_per_second=float(1 / values.mean()))


def run(trainer, dataset, reference, arrays, meta, repeats, warmup, rng_tools, logger, report, save):
    """First reproduce legacy DataLoader RNG; measured passes never load GT."""
    capture_rng, restore_rng = rng_tools
    rows = reference['per_image']
    states = []
    validation = new_guard()
    # Do NOT reseed here: model construction/loader initialization consume RNG
    # in the original evaluator. Python random.random() is also consumed by
    # BaseDataset.__getitem__ even when training=False.
    for i, items in enumerate(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)):
        if i >= len(rows):
            raise ValueError('Dataset longer than crop-count reference')
        row = rows[i]
        if (str(Path(items['img_path'][0]).resolve()) != str(Path(row['image']).resolve())
                or items['cls_name'][0] != row['class']):
            raise ValueError('Dataset order/identity differs from crop-count reference')
        states.append(capture_rng())
        prediction = predict_path(trainer, row['image'], row['class'], meta, 'baseline', rng_tools)
        check_prediction(prediction, row, arrays, 'baseline', validation)
        if (int(items['anomaly'][0]) != row['anomaly'] or not np.array_equal(
                items['img_mask'][0, 0].numpy() > .5, arrays[row['class']]['mask'][row['cache_position']])):
            raise ValueError('Dataset GT changed from crop-count reference')
        if (i + 1) % 64 == 0:
            logger.info('Untimed baseline/RNG replay %d/%d' % (i + 1, len(rows)))
    if len(states) != len(rows):
        raise ValueError('Dataset shorter than crop-count reference')
    validation['passed'] = True
    report['untimed_baseline_validation'] = validation
    logger.info('Untimed baseline/RNG replay PASS: %d images' % len(rows))
    save()
    for repeat in range(repeats):
        # Fixed rotation, not a best-order selection; reduce systematic drift.
        order = MODES[repeat % len(MODES):] + MODES[:repeat % len(MODES)]
        for mode in order:
            for i in range(warmup):
                index = i % len(rows)
                row = rows[index]
                restore_rng(states[index])
                predict_path(trainer, row['image'], row['class'], meta, mode, rng_tools)
            synchronize(trainer.device)
            guard, times, calls = new_guard(), [], 0
            for i, row in enumerate(rows):
                # Compatibility RNG restoration is outside the timing window.
                restore_rng(states[i])
                prediction, elapsed = timed_prediction(trainer, row['image'], row['class'], meta, mode, rng_tools)
                times.append(elapsed)
                calls += prediction['model_calls']
                # CPU/cache checks and logging are deliberately NOT timed.
                check_prediction(prediction, row, arrays, mode, guard)
                if (i + 1) % 128 == 0:
                    logger.info('Round %d %s %d/%d' % (repeat + 1, mode, i + 1, len(rows)))
            guard['passed'] = guard['checked_images'] == len(rows)
            entry = dict(round=repeat + 1, mode=mode, order=list(order), model_calls=calls,
                         equivalence=guard, latency=latency_summary(times), per_image_seconds=times)
            report['passes'].append(entry)
            logger.info('Round %d %s: mean=%.3f ms p95=%.3f ms calls=%d equivalence=PASS' % (
                repeat + 1, mode, entry['latency']['mean_ms'], entry['latency']['p95_ms'], calls))
            save()
    report['summary'] = {}
    for mode in MODES:
        passes = [p for p in report['passes'] if p['mode'] == mode]
        summary = latency_summary([t for p in passes for t in p['per_image_seconds']])
        summary['round_mean_ms'] = [p['latency']['mean_ms'] for p in passes]
        summary['round_mean_std_ms'] = float(np.std(summary['round_mean_ms']))
        summary['model_calls_per_image'] = CALLS[mode]
        summary['unique_images'] = len(rows)
        summary['images_note'] = 'images counts timed observations across repeated passes, not unique samples'
        report['summary'][mode] = summary
    base = report['summary']['baseline']['mean_ms']
    dual = report['summary']['adaptive']['mean_ms']
    for mode, summary in report['summary'].items():
        summary['latency_ratio_vs_baseline'] = summary['mean_ms'] / base
        summary['latency_reduction_vs_dual_percent'] = (1 - summary['mean_ms'] / dual) * 100
    report['all_reference_checks_passed'] = True


def main(args):
    source_path, source, baseline, output, source_reference, reference_path, reference, arrays = prepare(args)
    from dataset import get_data
    from method import AdaCLIP_Trainer
    from tools import Logger, setup_seed
    from tools.gate_equivalence import _rng_state, _restore_rng
    from tools.spatial_channel_pilot import import_baseline, frozen_guard

    meta = source['metadata']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda' and not args.allow_cpu:
        raise RuntimeError('CUDA is required for the GPU benchmark; --allow_cpu is only a CPU smoke check')
    setup_seed(meta['seed'])
    config = json.loads((Path(__file__).parent / 'model_configs' / (meta['model'] + '.json')).read_text())
    trainer = AdaCLIP_Trainer(backbone=meta['model'],
        feat_list=[config['vision_cfg']['layers'] // 4 * i for i in (1, 2, 3, 4)],
        input_dim=config['vision_cfg']['width'], output_dim=config['embed_dim'], learning_rate=0.,
        device=device, image_size=meta['image_size'], prompting_depth=meta['prompting_depth'],
        prompting_length=meta['prompting_length'], prompting_branch=meta['prompting_branch'],
        prompting_type=meta['prompting_type'], use_hsf=meta['use_hsf'], k_clusters=meta['k_clusters'], fusion_mode='add').to(device)
    before = import_baseline(trainer, str(baseline))
    for p in trainer.parameters():
        p.requires_grad_(False)
        p.grad = None
    trainer.eval()
    versions = {k: v._version for k, v in trainer.clip_model.state_dict().items()}
    classes, dataset, _ = get_data(dataset_type_list=args.testing_data, transform=trainer.preprocess,
                                  target_transform=trainer.transform, training=False)
    identities = [str((Path(dataset.root) / r['img_path']).resolve()) for r in dataset.data_all]
    source_ids = {str(Path(r['image']).resolve()) for split in ('train', 'validation')
                  for r in source_reference[split]['per_image']}
    if (getattr(dataset, 'training', True) or len(dataset) != len(identities)
            or set(classes) != set(reference['metrics']) or set(identities) & source_ids
            or identities != [str(Path(r['image']).resolve()) for r in reference['per_image']]):
        raise ValueError('Dataset identities/order/classes differ, training augmentation enabled, or source overlap')
    output.mkdir(parents=True, exist_ok=True)
    logger = Logger(str(output / 'standalone_latency.txt'))
    report = dict(status='running', testing_data=args.testing_data,
        protocol=dict(source_report=str(source_path),
            source_report_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            reference_eval=str(reference_path), reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            source_config={k: meta[k] for k in CONFIG_KEYS}, modes=list(MODES),
            model_calls_per_image=CALLS, repeats=args.repeats, warmup_per_mode_per_round=args.warmup,
            timing_scope='file decode + preprocess + H2D + global forward + held prompts + crop/align/fuse + sigma4 + CPU outputs',
            timing_excludes='model/checkpoint load, loader/GT, compatibility RNG reset, reference checks, warmup, logs/report I/O',
            synchronize='CUDA synchronization immediately before/after each whole image inference',
            filesystem_cache='warm/repeated reads; NOT a controlled cold-storage benchmark',
            independent_mode_passes=True, shared_global_or_crop_predictions=False,
            support='unchanged original dual-box intersection; single A uses B geometry ONLY',
            per_image_rng='saved from untimed legacy DataLoader baseline replay; no target-label decisions',
            no_training=True, no_target_tuning=True, no_early_exit=True,
            target_results_are_exploratory=True, single_a_fixed_primary=True,
            background_alignment_used=False, new_map_cache_bytes=0),
        environment=dict(device=device, gpu=torch.cuda.get_device_name() if device == 'cuda' else None,
            torch_version=str(torch.__version__), cuda_version=torch.version.cuda, cpu_threads=torch.get_num_threads(),
            gpu_benchmark=device == 'cuda'), passes=[])
    path = output / 'standalone_latency.json'

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')

    save()
    logger.info('Standalone benchmark: images=%d repeats=%d warmup=%d; baseline=1, single A=2, dual=3 forwards.' % (
        len(dataset), args.repeats, args.warmup))
    try:
        run(trainer, dataset, reference, arrays, meta, args.repeats, args.warmup,
            (_rng_state, _restore_rng), logger, report, save)
        report['unchanged_detector_tensors'] = frozen_guard(trainer, before, versions)
        report['status'] = 'complete'
        for mode, values in report['summary'].items():
            logger.info('%s mean=%.3f ms p95=%.3f ms ratio=%.3fx baseline' % (
                mode, values['mean_ms'], values['p95_ms'], values['latency_ratio_vs_baseline']))
        save()
        logger.info('Standalone latency DONE; all reference maps/scores PASS; frozen detector PASS')
    except Exception as error:
        report.update(status='failed', error=str(error))
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--source_report', required=True)
    parser.add_argument('--reference_eval', required=True, help='crop_count_eval.json AND its original cache directory')
    parser.add_argument('--testing_data', choices=('mpdd', 'visa', 'btad'), default='mpdd')
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--allow_cpu', action='store_true', help='CPU smoke check, not a GPU latency result')
    parser.set_defaults(random_seeds=[111, 222, 333])
    main(parser.parse_args())
