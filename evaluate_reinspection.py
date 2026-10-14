"""Full target evaluation with frozen, source-calibrated crop acceptance rules.

Four policies share the SAME two local predictions and keep the original global
HSF image score. Category-pooled pixel metrics use the existing official helpers.
Two disk-backed map caches avoid retaining four datasets of maps in RAM.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch

from calibrate_reinspection import accepts, match_initial_acceptance, CONFIG_KEYS, validate_report
from source_crop_pilot import capture_prompts, reuse_prompts
from source_reinspection_reliability import select_windows, align_view, evidence_statistics, candidate_updates


POLICIES = ('baseline', 'all_fusion', 'consistency', 'matched_score')


def load_fixed_rules(args):
    baseline, rules_path, report_path = map(Path, (args.baseline, args.calibration_rules, args.calibration_report))
    if not all(p.is_file() for p in (baseline, rules_path, report_path)):
        raise FileNotFoundError('Baseline, calibration rules and train-split report must exist')
    rules = json.loads(rules_path.read_text(encoding='utf8'))
    report = json.loads(report_path.read_text(encoding='utf8'))
    if rules.get('status') != 'complete' or report.get('status') != 'complete' or report['metadata'].get('split') != 'train':
        raise ValueError('Require completed rules and an explicitly marked train-split calibration report')
    if rules['calibration_report_sha256'] != hashlib.sha256(report_path.read_bytes()).hexdigest():
        raise ValueError('Calibration report contents differ from the report used to fit rules')
    if hashlib.sha256(baseline.read_bytes()).hexdigest() != rules['config']['baseline_sha256']:
        raise ValueError('Supplied detector checkpoint differs from source-calibrated baseline')
    for key in CONFIG_KEYS:
        if report['metadata'].get(key) != rules['config'].get(key):
            raise ValueError('Calibration report/rules configuration mismatch: ' + key)
    # Exact reference identities remain available in the original pilot on server.
    reference = json.loads(Path(report['metadata']['reference_report']).read_text(encoding='utf8'))
    validate_report(report, reference['train']['per_image'], 'calibration')
    if args.testing_data in report['metadata']['sources']:
        raise ValueError('Target dataset must differ from baseline auxiliary training sources')
    consistency = rules['rules']['consistency']['selected_rule']
    if consistency['feature'] != 'soft_iou' or consistency['kind'] not in ('threshold', 'reject_all', 'accept_all'):
        raise ValueError('Expected a calibrated soft_iou consistency rule')
    if consistency['kind'] == 'threshold' and not (isinstance(consistency['threshold'], (int, float))
            and math.isfinite(consistency['threshold']) and 0 <= consistency['threshold'] <= 1):
        raise ValueError('Invalid consistency threshold')
    matched = match_initial_acceptance(report['per_image'], consistency)
    return {'config': rules['config'], 'consistency': consistency, 'matched_score': matched['selected_rule'],
            'matching': matched, 'calibration_source': 'mvtec',
            'calibration_rules_path': str(rules_path.resolve()), 'calibration_report_path': str(report_path.resolve()),
            'thresholds_fixed_before_target_data_loading': True}


def policy_decisions(evidence, fixed):
    row = {'evidence': evidence}
    return {'baseline': False, 'all_fusion': bool(evidence['valid']),
            'consistency': accepts(row, fixed['consistency']),
            'matched_score': accepts(row, fixed['matched_score'])}


def inspect_image(trainer, image, cls_name, original, fixed, rng_tools, check=False):
    """Prediction-only function: no anomaly labels or masks are accepted."""
    capture_rng, restore_rng = rng_tools
    rng = capture_rng()
    device = trainer.device
    clip = trainer.clip_model

    def forward(tensor):
        restore_rng(rng)
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            amap, score = clip(tensor.to(device), cls_name, aggregation=True)
        if device == 'cuda':
            torch.cuda.synchronize()
        if not torch.isfinite(amap).all() or not torch.isfinite(score).all():
            raise RuntimeError('Non-finite target model output')
        return amap[0].float().cpu().numpy(), float(score[0]), time.perf_counter() - start

    base, image_score, baseline_seconds = forward(image)
    # All extra calls must leave the baseline RNG trajectory intact.
    after_baseline_rng = capture_rng()
    prompts = capture_prompts(clip)
    equivalence = None
    try:
        if check:
            with reuse_prompts(clip, prompts):
                held, held_score, _ = forward(image)
            if not np.allclose(base, held, atol=1e-6, rtol=1e-5) or not math.isclose(image_score, held_score, abs_tol=1e-6, rel_tol=1e-5):
                raise RuntimeError('Full-image held-prompt map/score equivalence failed')
            equivalence = {'passed': True, 'images': 1, 'max_map_delta': float(np.abs(base - held).max()),
                           'score_delta': abs(image_score - held_score)}
        config = fixed['config']
        selection = select_windows(base, original.size, config['crop_fraction'], config['shift_fraction'])
        views, supports, times = [], [], []
        for box in selection['boxes_native']:
            crop = trainer.preprocess(original.crop(box)).unsqueeze(0)
            with reuse_prompts(clip, prompts):
                values, _, elapsed = forward(crop)
            aligned, valid = align_view(values, box, original.size, base.shape, config['border_fraction'])
            views.append(aligned)
            supports.append(valid)
            times.append(elapsed)
        common = supports[0] & supports[1]
        if not selection['distinct_windows']:
            common[:] = False
        # Only probability consistency and INITIAL score are used by these rules.
        # Zero margin placeholders do not affect their computation or validity.
        z = np.zeros_like(base)
        evidence = evidence_statistics(base, *views, z, z, common, selection['peak_xy_grid'])
        evidence = {k: evidence[k] for k in ('valid', 'soft_iou', 'baseline_core_mean', 'shared_pixels')}
        decisions = policy_decisions(evidence, fixed)
        update_support = common if evidence['valid'] else np.zeros_like(common)
        fused = candidate_updates(base, *views, update_support)['two_crop_fixed_fusion']
        return {'baseline_map': gaussian_filter(base, sigma=4), 'fused_map': gaussian_filter(fused, sigma=4),
                'image_score': image_score, 'evidence': evidence, 'decisions': decisions,
                'selection': selection, 'baseline_seconds': baseline_seconds, 'crop_seconds': times,
                'initial_equivalence': equivalence}
    finally:
        restore_rng(after_baseline_rng)


def pooled_metrics(masks, baseline, fused, decisions, labels, scores, calculate_px, calculate_im):
    """Use official category-pooled metrics; every policy uses the GLOBAL score."""
    ap_im, auroc_im, f1_im = calculate_im(labels, scores)
    result = {}
    for policy in POLICIES:
        flags = np.asarray([d[policy] for d in decisions], dtype=bool)
        prediction = baseline if policy == 'baseline' else np.where(flags[:, None, None], fused, baseline)
        auroc_px, f1_px, ap_px = calculate_px(masks, prediction)
        values = dict(auroc_px=float(auroc_px), auroc_im=float(auroc_im), f1_px=float(f1_px),
                      f1_im=float(f1_im), ap_px=float(ap_px), ap_im=float(ap_im))
        if not all(math.isfinite(v) for v in values.values()):
            raise RuntimeError('Non-finite pooled metric')
        result[policy] = values
    return result


def main(args):
    output = Path(args.save_path).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Use a new/empty save_path; existing experiments are preserved')
    if any(output in Path(p).resolve().parents for p in (args.baseline, args.calibration_rules, args.calibration_report)):
        raise ValueError('Output directory must be separate from source experiment inputs')
    fixed = load_fixed_rules(args)
    from dataset import get_data
    from method import AdaCLIP_Trainer
    from tools import Logger, setup_seed, write2csv, calculate_average_metric
    from tools.metrics import calculate_px_metrics, calculate_im_metrics
    from tools.gate_equivalence import _rng_state, _restore_rng
    from tools.spatial_channel_pilot import import_baseline, frozen_guard

    meta = fixed['config']
    setup_seed(meta['seed'])
    config = json.loads((Path(__file__).parent / 'model_configs' / (meta['model'] + '.json')).read_text())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'csvs').mkdir()
    (output / 'cache').mkdir()
    logger = Logger(str(output / 'reinspection_eval.txt'))
    # Freeze/publish both rules before loading target dataset metadata or labels.
    (output / 'fixed_target_rules.json').write_text(json.dumps(fixed, indent=2, allow_nan=False), encoding='utf8')
    logger.info('FIXED SOURCE RULES: ' + json.dumps({k: fixed[k] for k in ('consistency', 'matched_score', 'matching')}))
    trainer = AdaCLIP_Trainer(backbone=meta['model'],
        feat_list=[config['vision_cfg']['layers'] // 4 * i for i in (1, 2, 3, 4)],
        input_dim=config['vision_cfg']['width'], output_dim=config['embed_dim'], learning_rate=0.,
        device=device, image_size=meta['image_size'], prompting_depth=meta['prompting_depth'],
        prompting_length=meta['prompting_length'], prompting_branch=meta['prompting_branch'], prompting_type=meta['prompting_type'],
        use_hsf=meta['use_hsf'], k_clusters=meta['k_clusters'], fusion_mode='add').to(device)
    before = import_baseline(trainer, args.baseline)
    for p in trainer.parameters():
        p.requires_grad_(False)
        p.grad = None
    trainer.eval()
    versions = {k: v._version for k, v in trainer.clip_model.state_dict().items()}
    logger.info('Strict baseline import PASS; detector tensors=%d' % len(before))
    classes, dataset, _ = get_data(dataset_type_list=args.testing_data, transform=trainer.preprocess,
                                  target_transform=trainer.transform, training=False)
    if getattr(dataset, 'training', True):
        raise ValueError('Target augmentation must be disabled')
    counts = {name: sum(r['cls_name'] == name for r in dataset.data_all) for name in classes}
    if not counts or any(n == 0 for n in counts.values()) or sum(counts.values()) != len(dataset):
        raise ValueError('Invalid target category counts')
    cache, class_rows = {}, {name: [] for name in classes}
    side = meta['image_size']
    disk_bytes = len(dataset) * side * side * 9  # Two float32 maps + one bool mask.
    logger.info('Target images=%d; map cache estimated %.2f GiB; image-level scores held at baseline.' %
                (len(dataset), disk_bytes / 1024 ** 3))
    for i, name in enumerate(classes):
        cache[name] = {}
        for key in ('baseline', 'fused', 'mask'):
            path = output / 'cache' / ('class_%02d_%s.npy' % (i, key))
            dtype = np.bool_ if key == 'mask' else np.float32
            cache[name][key] = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=(counts[name], side, side))
    report = {'status': 'running', 'testing_data': args.testing_data, 'fixed_rules': fixed,
              'image_score_policy': 'original global AdaCLIP/HSF score for every method',
              'metric_protocol': 'existing category-pooled helpers, then unweighted category average; sigma4 once after map fusion',
              'repeated_visa_evaluation_is_exploratory': args.testing_data == 'visa',
              'cache_estimated_bytes': disk_bytes, 'per_image': [], 'metrics': {}, 'acceptance': {}}
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    phase_start = time.perf_counter()
    for i, items in enumerate(loader):
        with Image.open(items['img_path'][0]) as image:
            original = image.convert('RGB')
        prediction = inspect_image(trainer, items['img'], items['cls_name'], original, fixed,
                                   (_rng_state, _restore_rng), check=i == 0)
        if i == 0:
            report['initial_equivalence'] = prediction['initial_equivalence']
            logger.info('Full-image held-prompt baseline equivalence PASS.')
        # GT and anomaly label are used ONLY after all policy decisions.
        name = items['cls_name'][0]
        position = len(class_rows[name])
        cache[name]['baseline'][position] = prediction['baseline_map']
        cache[name]['fused'][position] = prediction['fused_map']
        cache[name]['mask'][position] = items['img_mask'][0, 0].numpy() > .5
        row = {'class': name, 'image': items['img_path'][0], 'anomaly': int(items['anomaly'][0]),
               'cache_position': position, 'image_score': prediction['image_score'],
               'evidence': prediction['evidence'], 'decisions': prediction['decisions'],
               **prediction['selection'], 'baseline_seconds': prediction['baseline_seconds'],
               'crop_seconds': prediction['crop_seconds']}
        class_rows[name].append(row)
        report['per_image'].append(row)
        if (i + 1) % 32 == 0 or i + 1 == len(dataset):
            logger.info('Reinspection target progress=%d/%d elapsed=%.1fs' % (i + 1, len(dataset), time.perf_counter() - phase_start))
            (output / 'reinspection_eval.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    report['inference_loop_seconds'] = time.perf_counter() - phase_start
    report['unchanged_detector_tensors'] = frozen_guard(trainer, before, versions)
    for name in classes:
        for array in cache[name].values():
            array.flush()
        rows = class_rows[name]
        if len(rows) != counts[name]:
            raise RuntimeError('Missing target samples in category ' + name)
        logger.info('Computing pooled pixel metrics for category=' + name)
        metrics = pooled_metrics(cache[name]['mask'], cache[name]['baseline'], cache[name]['fused'],
            [r['decisions'] for r in rows], np.asarray([r['anomaly'] for r in rows]),
            np.asarray([r['image_score'] for r in rows]), calculate_px_metrics, calculate_im_metrics)
        report['metrics'][name] = metrics
        for policy, values in metrics.items():
            path = output / 'csvs' / ('%s_reinspection_%s.csv' % (args.testing_data, policy))
            write2csv(values, list(classes) + ['Average'], name, str(path))
    report['average'] = {p: calculate_average_metric({name: report['metrics'][name][p] for name in classes}) for p in POLICIES}
    for policy, values in report['average'].items():
        write2csv(values, list(classes) + ['Average'], 'Average',
                  str(output / 'csvs' / ('%s_reinspection_%s.csv' % (args.testing_data, policy))))
        report['acceptance'][policy] = {'images': len(dataset), 'accepted': sum(r['decisions'][policy] for r in report['per_image']),
            'normal_images': sum(r['anomaly'] == 0 for r in report['per_image']),
            'accepted_normal': sum(r['decisions'][policy] for r in report['per_image'] if r['anomaly'] == 0),
            'anomalous_images': sum(r['anomaly'] == 1 for r in report['per_image']),
            'accepted_anomalous': sum(r['decisions'][policy] for r in report['per_image'] if r['anomaly'] == 1)}
        logger.info('%s Average I-Auroc:%.2f I-F1:%.2f I-AP:%.2f P-Auroc:%.2f P-F1:%.2f P-AP:%.2f accepted=%d/%d' % (
            policy, values['auroc_im'], values['f1_im'], values['ap_im'], values['auroc_px'], values['f1_px'], values['ap_px'],
            report['acceptance'][policy]['accepted'], len(dataset)))
    report['timing'] = {'baseline_forward_seconds': sum(r['baseline_seconds'] for r in report['per_image']),
                        'two_crop_forward_seconds': sum(sum(r['crop_seconds']) for r in report['per_image']),
                        'all_methods_share_two_crops_no_policy_compute_savings_claim': True}
    report['status'] = 'complete'
    (output / 'reinspection_eval.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    logger.info('Reinspection target evaluation DONE; all four methods complete; frozen detector PASS.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--calibration_rules', required=True)
    parser.add_argument('--calibration_report', required=True)
    parser.add_argument('--testing_data', choices=('visa', 'mpdd', 'btad'), default='visa')
    parser.add_argument('--save_path', required=True)
    main(parser.parse_args())
