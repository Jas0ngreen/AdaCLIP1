"""Source-only matched-budget crop-selection ablation. No training or gating.

All variants use two equal-size shifted crops and the existing fixed 0.5 fusion.
Random seeds are independent repetitions, never an ensemble or best-seed search.
Metrics are per-image diagnostics, not category-pooled target benchmark scores.
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

from source_crop_pilot import CONFIG_KEYS, capture_prompts, reuse_prompts, region_metrics, validate
from source_reinspection_reliability import (
    select_windows, windows_at_grid_anchor, align_view, evidence_statistics, candidate_updates)


METRICS = ('pixel_ap', 'pixel_auroc', 'background_mean', 'background_q99')


def average(values):
    return float(np.mean(values)) if values else None


def selections(base, native_size, config, identity, random_seeds):
    """No GT/label input; local random generators cannot disturb CLIP/HSF RNG."""
    shape = base.shape
    h, w = shape
    crop, shift = config['crop_fraction'], config['shift_fraction']
    result = {'adaptive': select_windows(base, native_size, crop, shift),
              'fixed_center': windows_at_grid_anchor((w // 2, h // 2), shape, native_size, crop, shift)}
    for seed in random_seeds:
        digest = hashlib.sha256(('%d:%s' % (seed, identity)).encode('utf8')).digest()
        rng = np.random.RandomState(int.from_bytes(digest[:4], 'big'))
        anchor = (int(rng.randint(w)), int(rng.randint(h)))
        result['random_%d' % seed] = windows_at_grid_anchor(anchor, shape, native_size, crop, shift)
    return result


def inspect_image(trainer, image, cls_name, original, config, identity, random_seeds, rng_tools, check=False):
    """Selection/inference must finish before labels or GT enter the pipeline."""
    return _inspect_image(trainer, image, cls_name, original, config, identity, random_seeds,
                          rng_tools, check, crop_count=False)


def inspect_crop_count(trainer, image, cls_name, original, config, identity, random_seeds, rng_tools, check=False):
    """Matched-support single A/B vs dual; only the adaptive pair is inferred.

    Both boxes define the SAME geometric support for all three variants. Their
    safety check needs no crop-B prediction, even though this shared experiment
    computes B to evaluate the dual variant. A is the prespecified single control.
    """
    return _inspect_image(trainer, image, cls_name, original, config, identity, random_seeds,
                          rng_tools, check, crop_count=True)


def _inspect_image(trainer, image, cls_name, original, config, identity, random_seeds, rng_tools, check, crop_count):
    capture_rng, restore_rng = rng_tools
    rng = capture_rng()
    device, clip = trainer.device, trainer.clip_model

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
            raise RuntimeError('Non-finite model prediction')
        values = amap[0].float().cpu().numpy()
        if values.ndim != 2 or np.min(values) < 0 or np.max(values) > 1:
            raise ValueError('Expected a 2D probability map')
        return values, float(score[0]), time.perf_counter() - start

    base, score, seconds = forward(image)
    after_baseline_rng = capture_rng()
    prompts = capture_prompts(clip)
    try:
        equivalence = None
        if check:
            with reuse_prompts(clip, prompts):
                held, held_score, _ = forward(image)
            if not np.allclose(base, held, atol=1e-6, rtol=1e-5) or not math.isclose(score, held_score, abs_tol=1e-6, rel_tol=1e-5):
                raise RuntimeError('Full-image held-prompt equivalence failed')
            equivalence = dict(passed=True, max_map_delta=float(np.abs(base - held).max()),
                               score_delta=abs(score - held_score))
        choices = ({'adaptive': select_windows(base, original.size, config['crop_fraction'], config['shift_fraction'])}
                   if crop_count else selections(base, original.size, config, identity, random_seeds))
        maps, supports, records = {'baseline': gaussian_filter(base, sigma=4)}, {}, {}
        for mode, selection in choices.items():
            views, valid_masks, times = [], [], []
            for box in selection['boxes_native']:
                crop = trainer.preprocess(original.crop(box)).unsqueeze(0)
                with reuse_prompts(clip, prompts):
                    values, _, elapsed = forward(crop)
                aligned, valid = align_view(values, box, original.size, base.shape, config['border_fraction'])
                views.append(aligned)
                valid_masks.append(valid)
                times.append(elapsed)
            common = valid_masks[0] & valid_masks[1]
            if not selection['distinct_windows']:
                common[:] = False
            z = np.zeros_like(base)
            # Identical geometry-only safety test for each selector's OWN anchor.
            # Neither soft_iou nor anomaly confidence decides whether to update.
            evidence = (evidence_statistics(base, z, z, z, z, common, selection['peak_xy_grid']) if crop_count else
                        evidence_statistics(base, *views, z, z, common, selection['peak_xy_grid']))
            support = common if evidence['valid'] else np.zeros_like(common)
            updates = candidate_updates(base, *views, support, weight=.5)
            fused = updates['two_crop_fixed_fusion']
            maps[mode] = gaussian_filter(fused, sigma=4)
            supports[mode] = support
            records[mode] = dict(anchor_xy_grid=selection['peak_xy_grid'],
                boxes_native=selection['boxes_native'], distinct_windows=selection['distinct_windows'],
                actual_origin_shift_native=selection['actual_origin_shift_native'],
                geometry_valid=bool(evidence['valid']), shared_pixels=int(common.sum()),
                update_pixels=int(support.sum()), update_fraction=float(support.mean()),
                crop_seconds=times)
            if crop_count:
                single_b = base.copy()
                single_b[support] = .5 * base[support] + .5 * views[1][support]
                for name, values, index in (('single_a', updates['single_crop_fixed_fusion'], 0),
                                             ('single_b', single_b, 1)):
                    maps[name] = gaussian_filter(values, sigma=4)
                    supports[name] = support
                    records[name] = {**records[mode], 'crop_seconds': [times[index]],
                                     'forward_box_native': selection['boxes_native'][index],
                                     'matched_support_uses_both_box_geometries': True}
        return dict(maps=maps, supports=supports, selections=records, image_score=score,
                    baseline_seconds=seconds, initial_equivalence=equivalence)
    finally:
        restore_rng(after_baseline_rng)


def macro(rows, mode, metric):
    groups = sorted({(r['source'], r['class']) for r in rows})
    values = []
    for source, category in groups:
        part = [r['metrics'][mode][metric] for r in rows
                if r['source'] == source and r['class'] == category and r['metrics'][mode][metric] is not None]
        if part:
            values.append(average(part))
    return average(values)


def summarize(rows, modes):
    """Never treat random repeats as independently sampled images."""
    abnormal, normal = ([r for r in rows if r['anomaly'] == label] for label in (1, 0))
    output = dict(images=len(rows), anomalous_images=len(abnormal), normal_images=len(normal), modes={}, comparisons={})
    for mode in modes:
        entry = {}
        for metric in METRICS:
            group = abnormal if metric in ('pixel_ap', 'pixel_auroc') else normal
            value, baseline = macro(group, mode, metric), macro(group, 'baseline', metric)
            entry[metric] = dict(class_balanced=value,
                gain_vs_baseline=value - baseline if value is not None and baseline is not None else None,
                defined_images=sum(r['metrics'][mode][metric] is not None for r in group))
        if mode not in ('baseline', 'random_mean'):
            entry['geometry_valid_images'] = sum(r['selections'][mode]['geometry_valid'] for r in rows)
            entry['mean_update_fraction'] = average([r['selections'][mode]['update_fraction'] for r in rows])
            entry['mean_crop_forward_seconds'] = average([sum(r['selections'][mode]['crop_seconds']) for r in rows])
            entry['mean_defect_coverage_in_update_support'] = average([
                r['defect_coverage'][mode] for r in abnormal if r['defect_coverage'][mode] is not None])
        output['modes'][mode] = entry
    for control in ('fixed_center', 'random_mean'):
        pairs = [(r['metrics']['adaptive']['pixel_ap'], r['metrics'][control]['pixel_ap']) for r in abnormal]
        deltas = [a - b for a, b in pairs if a is not None and b is not None]
        a, b = macro(abnormal, 'adaptive', 'pixel_ap'), macro(abnormal, control, 'pixel_ap')
        output['comparisons']['adaptive_vs_' + control] = dict(
            class_balanced_ap_difference=a - b if a is not None and b is not None else None,
            image_average_ap_difference=average(deltas), defined_images=len(deltas),
            improved_images=sum(d > 1e-4 for d in deltas), worsened_images=sum(d < -1e-4 for d in deltas),
            near_unchanged_images=sum(abs(d) <= 1e-4 for d in deltas))
    return output


def prepare(args):
    source_path = Path(args.source_report).resolve()
    source = json.loads(source_path.read_text(encoding='utf8'))
    if source.get('status') != 'complete' or source.get('unchanged_detector_tensors', 0) <= 0:
        raise ValueError('Require completed source reliability report with frozen-detector PASS')
    meta = source['metadata']
    if not meta.get('no_target_evaluation') or meta['sources'] != ['mvtec', 'colondb']:
        raise ValueError('Only the original mvtec+colondb source report is allowed')
    if (not 0 < meta['crop_fraction'] < 1 or not 0 < meta['shift_fraction'] < meta['crop_fraction'] / 4
            or not 0 <= meta['border_fraction'] < .2 or meta['candidate_fusion_weight'] != .5):
        raise ValueError('Unsupported source crop geometry/fusion configuration')
    if len(args.random_seeds) != len(set(args.random_seeds)) or not args.random_seeds or any(s < 0 for s in args.random_seeds):
        raise ValueError('Random seeds must be distinct nonnegative integers')
    # Reuse original reference validation without changing its public interface.
    args.reference_report = meta['reference_report']
    args.crop_fraction = meta['crop_fraction']
    baseline, reference, out, reference_data = validate(args)
    if out in source_path.parents or source_path.parent in out.parents:
        raise ValueError('Output must be separate from source-report experiment directory')
    if hashlib.sha256(baseline.read_bytes()).hexdigest() != meta['baseline_sha256']:
        raise ValueError('Baseline checkpoint differs from source report')
    for key in CONFIG_KEYS:
        if meta[key] != reference_data['metadata'][key]:
            raise ValueError('Source/reference configuration mismatch: ' + key)
    args.seed = meta['seed']
    return source_path, source, baseline, reference, out, reference_data


def main(args):
    source_path, source, baseline, reference, out, reference_data = prepare(args)
    from dataset import get_data
    from method import AdaCLIP_Trainer
    from tools import Logger, setup_seed
    from tools.gate_equivalence import _rng_state, _restore_rng
    from tools.spatial_channel_pilot import import_baseline, make_loaders, frozen_guard

    meta = source['metadata']
    setup_seed(args.seed)
    config = json.loads((Path(__file__).parent / 'model_configs' / (meta['model'] + '.json')).read_text())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    loaders, manifest = make_loaders(trainer, args, reference_data, get_data)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'cache').mkdir()
    logger = Logger(str(out / 'source_selection_control.txt'))
    modes = ['baseline', 'adaptive', 'fixed_center'] + ['random_%d' % s for s in args.random_seeds]
    report = dict(status='running', metadata=dict(
        source_report=str(source_path), source_report_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        reference_report=str(reference), reference_report_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        detector_config={k: meta[k] for k in CONFIG_KEYS}, baseline_checkpoint=str(baseline),
        baseline_sha256=meta['baseline_sha256'], split=args.split, seed=args.seed,
        crop_fraction=meta['crop_fraction'], shift_fraction=meta['shift_fraction'], border_fraction=meta['border_fraction'],
        fusion_weight=.5, random_seeds=args.random_seeds, crops_per_variant=2,
        candidate_selectors='adaptive sigma4 argmax; fixed central grid pixel; uniform random grid pixel',
        random_identity='SHA256(seed:source:absolute_image_path), first 32 bits -> numpy RandomState',
        no_score_gate=True, geometry_safety='distinct shifted crops; own anchor/core/ring covered by common support',
        held_prompts='both visual and textual dynamic prompts from the same global forward',
        metric_units='0..1; per-image grid metrics, sigma4 once after fusion',
        random_mean='mean of per-image metrics across seeds; NOT AP of averaged maps or an ensemble',
        source_experiment_not_unseen_target_test=True, no_training=True, no_target_evaluation=True,
        no_threshold_tuning=True, per_image_model_calls_excluding_initial_check=1 + 2 * (len(modes) - 1)),
        per_image=[])
    report_path = out / 'source_selection_control.json'
    logger.info('SOURCE SELECTION CONTROL: split=%s images=%d seeds=%s; 2 crop forwards per variant.' %
                (args.split, len(manifest[args.split]), args.random_seeds))
    for source_name, loader in loaders[args.split]:
        for items in loader:
            setup_seed(args.seed + len(report['per_image']))
            identity = source_name + ':' + str(Path(items['img_path'][0]).resolve())
            with Image.open(items['img_path'][0]) as image:
                original = image.convert('RGB')
            prediction = inspect_image(trainer, items['img'], items['cls_name'], original, meta, identity,
                                       args.random_seeds, (_rng_state, _restore_rng), check=not report['per_image'])
            if not report['per_image']:
                report['initial_equivalence'] = prediction['initial_equivalence']
            # Labels/masks are consumed ONLY after every selection and forward.
            mask = items['img_mask'][0, 0].numpy() > .5
            metrics = {mode: region_metrics(mask, values) for mode, values in prediction['maps'].items()}
            metrics['random_mean'] = {key: average([metrics['random_%d' % s][key] for s in args.random_seeds
                                         if metrics['random_%d' % s][key] is not None]) for key in metrics['baseline']}
            row = dict(source=source_name, **{'class': items['cls_name'][0]}, image=items['img_path'][0],
                anomaly=int(items['anomaly'][0]), selections=prediction['selections'], metrics=metrics,
                image_score=prediction['image_score'], baseline_seconds=prediction['baseline_seconds'],
                defect_coverage={mode: float(mask[support].sum() / mask.sum()) if mask.any() else None
                                 for mode, support in prediction['supports'].items()})
            stem = '%03d' % (len(report['per_image']) + 1)
            row['cache'] = 'cache/' + stem + '.npz'
            np.savez_compressed(out / row['cache'], **prediction['maps'], mask=mask,
                                **{'support_' + k: v for k, v in prediction['supports'].items()})
            report['per_image'].append(row)
            logger.info('%s source=%s class=%s AP=%s' % (stem, source_name, row['class'],
                {k: v['pixel_ap'] for k, v in metrics.items()}))
            report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    expected = {(r['source'], str(Path(r['image']).resolve())) for r in manifest[args.split]}
    actual = [(r['source'], str(Path(r['image']).resolve())) for r in report['per_image']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise RuntimeError('Selected split identity mismatch')
    report['unchanged_detector_tensors'] = frozen_guard(trainer, before, versions)
    all_modes = modes + ['random_mean']
    report['by_source'] = {name: summarize([r for r in report['per_image'] if r['source'] == name], all_modes)
                           for name in meta['sources']}
    report['by_class'] = {name: {c: summarize([r for r in report['per_image'] if r['source'] == name and r['class'] == c], all_modes)
        for c in sorted({r['class'] for r in report['per_image'] if r['source'] == name})} for name in meta['sources']}
    report['status'] = 'complete'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    for name, summary in report['by_source'].items():
        logger.info(name + ' class-balanced per-image metrics: ' + json.dumps(summary, allow_nan=False))
    logger.info('Source selection control DONE; frozen detector PASS; report=' + str(report_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--source_report', required=True, help='Completed source_reinspection_reliability.json')
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--split', choices=('train', 'validation'), default='validation')
    parser.add_argument('--random_seeds', nargs='+', type=int, default=[111, 222, 333])
    main(parser.parse_args())
