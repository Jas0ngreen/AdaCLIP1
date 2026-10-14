"""Source-only diagnostic: can crop consistency/contrast predict useful updates?

Candidate selection and evidence statistics never consume labels or GT masks.
GT is used only to evaluate fixed candidate updates. No score fitting or training.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import functional as F

from source_crop_pilot import (BICUBIC, CONFIG_KEYS, capture_prompts, reuse_prompts,
                               region_metrics, validate)


def centered_box(cx, cy, width, height, image_width, image_height):
    x = min(max(round(cx - width / 2), 0), image_width - width)
    y = min(max(round(cy - height / 2), 0), image_height - height)
    return int(x), int(y), int(x + width), int(y + height)


def select_windows(baseline, native_size, crop_fraction=.5, shift_fraction=.05):
    """Same deterministic prediction-only selector for ALL images."""
    h, w = baseline.shape
    py, px = np.unravel_index(np.argmax(gaussian_filter(baseline, sigma=4)), baseline.shape)
    nw, nh = native_size
    cx, cy = (px + .5) * nw / w, (py + .5) * nh / h
    bw, bh = max(2, round(nw * crop_fraction)), max(2, round(nh * crop_fraction))
    sx, sy = max(1, round(nw * shift_fraction)), max(1, round(nh * shift_fraction))
    a = centered_box(cx - sx, cy - sy, bw, bh, nw, nh)
    b = centered_box(cx + sx, cy + sy, bw, bh, nw, nh)
    if a == b:
        # At corners both requested shifts can clamp to the same box.
        x, y, _, _ = b
        dx = sx if x + bw < nw else -sx
        dy = sy if y + bh < nh else -sy
        x, y = min(max(x + dx, 0), nw - bw), min(max(y + dy, 0), nh - bh)
        b = (x, y, x + bw, y + bh)
    return {'peak_xy_grid': [int(px), int(py)], 'boxes_native': [list(a), list(b)],
            'actual_origin_shift_native': [int(b[0] - a[0]), int(b[1] - a[1])],
            'distinct_windows': a != b}


def align_view(values, box, native_size, grid_size, border_fraction=.1):
    """Sample at global pixel centers; no rounding crop boxes to a second grid.

    Internal crop edges are excluded. Physical image edges remain valid.
    align_corners=False matches resize pixel-center geometry.
    """
    nw, nh = native_size
    gh, gw = grid_size
    x0, y0, x1, y1 = box
    x = (np.arange(gw, dtype=np.float32) + .5) * nw / gw
    y = (np.arange(gh, dtype=np.float32) + .5) * nh / gh
    gx, gy = np.meshgrid(2 * (x - x0) / (x1 - x0) - 1, 2 * (y - y0) / (y1 - y0) - 1)
    coords = torch.from_numpy(np.stack((gx, gy), axis=-1)).unsqueeze(0)
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32).copy())[None, None]
    aligned = F.grid_sample(tensor, coords, mode='bilinear', padding_mode='border',
                            align_corners=False)[0, 0].numpy()
    bx, by = border_fraction * (x1 - x0), border_fraction * (y1 - y0)
    lo_x, hi_x = x0 + (bx if x0 > 0 else 0), x1 - (bx if x1 < nw else 0)
    lo_y, hi_y = y0 + (by if y0 > 0 else 0), y1 - (by if y1 < nh else 0)
    valid = (x[None, :] >= lo_x) & (x[None, :] < hi_x) & (y[:, None] >= lo_y) & (y[:, None] < hi_y)
    return aligned, valid


@contextmanager
def capture_similarity_margin(model):
    """Observe projected normalized features; preserve original model outputs.

    FP32 cosine difference is taken BEFORE softmax, averaged over all layers.
    The implementation deliberately does not invert saturated FP16 probabilities.
    """
    key = 'visual_text_similarity'
    had_instance = key in model.__dict__
    previous = getattr(model, key)
    captured = {}

    def observe(image_feature, patch_token, text_feature, aggregation):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            maps = []
            for tokens in patch_token:
                similarities = tokens.float() @ text_feature.float()
                if similarities.shape[0] != 1 or similarities.shape[-1] != 2:
                    raise ValueError('Margin observer requires one image and normal/abnormal text columns')
                side = math.isqrt(similarities.shape[1])
                if side * side != similarities.shape[1]:
                    raise ValueError('Non-square patch grid')
                margin = (similarities[..., 1] - similarities[..., 0]).reshape(1, 1, side, side)
                maps.append(F.interpolate(margin, size=model.image_size, mode='bilinear', align_corners=True))
            captured['margin'] = torch.stack(maps).mean(0)[0, 0].cpu().numpy()
        return previous(image_feature, patch_token, text_feature, aggregation)

    try:
        setattr(model, key, observe)
        yield captured
    finally:
        if had_instance:
            setattr(model, key, previous)
        else:
            delattr(model, key)


def evidence_statistics(baseline, a, b, da, db, common, peak):
    """No GT: fixed core/ring from the INITIAL prediction, not crop maxima."""
    h, w = baseline.shape
    px, py = peak
    core_box = centered_box(px + .5, py + .5, max(2, round(w / 8)), max(2, round(h / 8)), w, h)
    outer_box = centered_box(px + .5, py + .5, max(3, round(w / 4)), max(3, round(h / 4)), w, h)
    core, outer = np.zeros_like(common), np.zeros_like(common)
    x0, y0, x1, y1 = core_box
    core[y0:y1, x0:x1] = True
    x0, y0, x1, y1 = outer_box
    outer[y0:y1, x0:x1] = True
    ring = outer & ~core & common
    core &= common
    if not common[py, px] or not core.any() or not ring.any():
        return {'valid': False, 'reason': 'Candidate/core/ring not jointly covered after boundary exclusion',
                'soft_iou': None, 'iou_at_fixed_0.5': None, 'mean_absolute_view_difference': None,
                'contrast_a': None, 'contrast_b': None, 'contrast_min': None, 'contrast_mean': None,
                'baseline_core_mean': None, 'local_core_mean': None,
                'shared_pixels': int(common.sum()), 'core_pixels': int(core.sum()), 'ring_pixels': int(ring.sum())}
    av, bv = a[common], b[common]
    denominator = float(np.maximum(av, bv).sum())
    # Weak, equal background maps must not be interpreted as strong evidence.
    soft_iou = float(np.minimum(av, bv).sum() / denominator) if float(np.maximum(av, bv).mean()) > 1e-6 else 0.
    active_a, active_b = av >= .5, bv >= .5
    union = np.count_nonzero(active_a | active_b)
    threshold_iou = float(np.count_nonzero(active_a & active_b) / union) if union else None
    ca = float(da[core].mean() - da[ring].mean())
    cb = float(db[core].mean() - db[ring].mean())
    return {'valid': True, 'soft_iou': soft_iou, 'iou_at_fixed_0.5': threshold_iou,
            'mean_absolute_view_difference': float(np.abs(av - bv).mean()),
            'contrast_a': ca, 'contrast_b': cb, 'contrast_min': min(ca, cb),
            'contrast_mean': (ca + cb) / 2,
            'baseline_core_mean': float(baseline[core].mean()),
            'local_core_mean': float(((a + b) / 2)[core].mean()),
            'shared_pixels': int(common.sum()), 'core_pixels': int(core.sum()), 'ring_pixels': int(ring.sum()),
            'core_box_grid': list(core_box), 'outer_box_grid': list(outer_box)}


def candidate_updates(baseline, a, b, common, weight=.5):
    """Matched support/weight; only number of crop predictions differs."""
    updates = {'baseline': baseline.copy()}
    for name, proposal in (('single_crop_fixed_fusion', a), ('two_crop_fixed_fusion', (a + b) / 2)):
        values = baseline.copy()
        values[common] = (1 - weight) * baseline[common] + weight * proposal[common]
        updates[name] = values
    return updates


def rank_analysis(rows):
    """Exploratory associations only; no fitting or picking acceptance thresholds."""
    eligible = [r for r in rows if r['evidence']['valid']]
    output = {'images': len(rows), 'valid_evidence_images': len(eligible),
              'anomalous_images_with_defined_ap': 0, 'predictors': {}}
    abnormal = [r for r in eligible if r['outcomes']['two_crop_ap_delta'] is not None]
    output['anomalous_images_with_defined_ap'] = len(abnormal)
    normal = [r for r in eligible if r['anomaly'] == 0]
    for key in ('soft_iou', 'contrast_min', 'contrast_mean', 'baseline_core_mean', 'local_core_mean'):
        info = {}
        for label, subset, outcome in (('anomalous_ap_gain', abnormal, 'two_crop_ap_delta'),
                                       ('normal_background_increase', normal, 'two_crop_background_delta')):
            x = np.asarray([r['evidence'][key] for r in subset])
            y = np.asarray([r['outcomes'][outcome] for r in subset])
            defined = len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0
            info[label] = {'images': len(x), 'spearman': float(spearmanr(x, y).correlation) if defined else None}
            if label == 'anomalous_ap_gain':
                positive = y > 1e-4
                info[label]['positive_ap_gain_images'] = int(positive.sum())
                info[label]['auroc_for_gain_over_0.01_percentage_point'] = (
                    float(roc_auc_score(positive, x)) if len(x) and positive.any() and (~positive).any() else None)
        output['predictors'][key] = info
    for name in ('baseline', 'single_crop_fixed_fusion', 'two_crop_fixed_fusion'):
        output[name] = {}
        for metric in ('pixel_ap', 'pixel_auroc', 'background_mean', 'background_q99'):
            vals = [r['metrics'][name][metric] for r in rows if r['metrics'][name][metric] is not None]
            output[name][metric] = {'mean': float(np.mean(vals)) if vals else None, 'valid_images': len(vals)}
    return output


def main(args):
    if (not 0 < args.crop_fraction < 1 or not 0 < args.shift_fraction < args.crop_fraction / 4
            or not 0 <= args.border_fraction < .2):
        raise ValueError('Require crop in (0,1), shift in (0,crop/4), border in [0,.2)')
    baseline, reference, out, reference_data = validate(args)
    from dataset import get_data
    from method import AdaCLIP_Trainer
    from tools import Logger, setup_seed
    from tools.gate_equivalence import _rng_state, _restore_rng
    from tools.spatial_channel_pilot import import_baseline, make_loaders, frozen_guard

    meta = reference_data['metadata']
    setup_seed(args.seed)
    config = json.loads((Path(__file__).parent / 'model_configs' / (meta['model'] + '.json')).read_text())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'cache').mkdir()
    logger = Logger(str(out / 'source_reinspection_reliability.txt'))
    logger.info('SOURCE RELIABILITY DIAGNOSTIC: automatic prediction-only windows; fixed global prompts; no training.')
    trainer = AdaCLIP_Trainer(backbone=meta['model'],
        feat_list=[config['vision_cfg']['layers'] // 4 * i for i in (1, 2, 3, 4)],
        input_dim=config['vision_cfg']['width'], output_dim=config['embed_dim'], learning_rate=0.,
        device=device, image_size=meta['image_size'], prompting_depth=meta['prompting_depth'],
        prompting_length=meta['prompting_length'], prompting_branch='VL', prompting_type='SD',
        use_hsf=meta['use_hsf'], k_clusters=meta['k_clusters'], fusion_mode='add').to(device)
    before = import_baseline(trainer, str(baseline))
    for p in trainer.parameters():
        p.requires_grad_(False)
        p.grad = None
    trainer.eval()
    clip = trainer.clip_model
    versions = {k: v._version for k, v in clip.state_dict().items()}
    loaders, manifest = make_loaders(trainer, args, reference_data, get_data)
    result = {'status': 'running', 'metadata': {**{k: meta[k] for k in CONFIG_KEYS},
        'baseline_checkpoint': str(baseline), 'reference_report': str(reference),
        'baseline_sha256': hashlib.sha256(baseline.read_bytes()).hexdigest(),
        'seed': args.seed, 'crop_fraction': args.crop_fraction, 'shift_fraction': args.shift_fraction,
        'border_fraction': args.border_fraction, 'sources': meta['sources'],
        'candidate_selector': 'argmax of initial map smoothed with sigma4; no GT/label input',
        'prompts': 'hold BOTH visual/text dynamic prompts from full image',
        'contrast': 'FP32 pre-softmax cosine difference, layer mean; fixed baseline core and ring',
        'core_side_fraction': .125, 'outer_side_fraction': .25,
        'metric_units': '0..1', 'candidate_fusion_weight': .5,
        'metrics': 'per-image full 518 grid; sigma4 applied after each candidate update',
        'score_combination': 'none', 'threshold_selection': 'none',
        'invalid_evidence_policy': 'preserve baseline; exclude from predictor rank analyses',
        'normal_false_positive_threshold': 'none; response changes are reported',
        'source_validation_is_not_unseen_to_baseline': True,
        'reused_validation_is_exploratory_not_an_independent_final_test': True,
        'no_target_evaluation': True}, 'per_image': []}
    output = out / 'source_reinspection_reliability.json'

    def forward(image, cls_name, rng, observe=True):
        _restore_rng(rng)
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            if observe:
                with capture_similarity_margin(clip) as capture:
                    amap, score = clip(image.to(device), cls_name, aggregation=True)
                margin = capture['margin']
            else:
                amap, score = clip(image.to(device), cls_name, aggregation=True)
                margin = None
        if device == 'cuda':
            torch.cuda.synchronize()
        if not torch.isfinite(amap).all() or not torch.isfinite(score).all() or (observe and not np.isfinite(margin).all()):
            raise RuntimeError('Non-finite forward output')
        return amap[0].float().cpu().numpy(), margin, time.perf_counter() - start

    for source, loader in loaders['validation']:
        for items in loader:
            setup_seed(args.seed + len(result['per_image']))
            rng = _rng_state()
            base, base_margin, seconds = forward(items['img'], items['cls_name'], rng)
            prompts = capture_prompts(clip)
            if not result['per_image']:
                plain, _, _ = forward(items['img'], items['cls_name'], rng, observe=False)
                with reuse_prompts(clip, prompts):
                    held, _, _ = forward(items['img'], items['cls_name'], rng)
                if not all(np.allclose(base, a, atol=1e-6, rtol=1e-5) for a in (plain, held)):
                    raise RuntimeError('Observer / held-prompt baseline equivalence failed')
                result['initial_equivalence'] = {'images': 1, 'passed': True,
                    'observer_max_map_delta': float(np.abs(base - plain).max()),
                    'held_prompt_max_map_delta': float(np.abs(base - held).max())}
                logger.info('Observer and full-image held-prompt equivalence PASS.')
            with Image.open(items['img_path'][0]) as image:
                original = image.convert('RGB')
            selection = select_windows(base, original.size, args.crop_fraction, args.shift_fraction)
            views, margins, supports, crop_seconds = [], [], [], []
            for box in selection['boxes_native']:
                image = trainer.preprocess(original.crop(box)).unsqueeze(0)
                with reuse_prompts(clip, prompts):
                    values, margin, elapsed = forward(image, items['cls_name'], rng)
                aligned, valid = align_view(values, box, original.size, base.shape, args.border_fraction)
                aligned_margin, _ = align_view(margin, box, original.size, base.shape, args.border_fraction)
                views.append(aligned)
                margins.append(aligned_margin)
                supports.append(valid)
                crop_seconds.append(elapsed)
            common = supports[0] & supports[1]
            if not selection['distinct_windows']:
                common[:] = False
            evidence = evidence_statistics(base, *views, *margins, common, selection['peak_xy_grid'])
            # Invalid perturbation evidence preserves baseline; excluded from rank analyses.
            update_support = common if evidence['valid'] else np.zeros_like(common)
            updates = candidate_updates(base, *views, update_support)
            # Labels/GT enter ONLY below, after all selection/evidence computations.
            mask = items['img_mask'][0, 0].numpy() > .5
            metrics = {name: region_metrics(mask, gaussian_filter(values, sigma=4)) for name, values in updates.items()}
            def delta(name, key):
                a, b = metrics[name][key], metrics['baseline'][key]
                return a - b if a is not None and b is not None else None
            row = {'source': source, 'class': items['cls_name'][0], 'image': items['img_path'][0],
                'anomaly': int(items['anomaly'][0]), 'native_size': list(original.size), **selection,
                'evidence': evidence, 'metrics': metrics,
                'outcomes': {'two_crop_ap_delta': delta('two_crop_fixed_fusion', 'pixel_ap'),
                             'single_crop_ap_delta': delta('single_crop_fixed_fusion', 'pixel_ap'),
                             'two_crop_background_delta': delta('two_crop_fixed_fusion', 'background_mean'),
                             'two_crop_background_q99_delta': delta('two_crop_fixed_fusion', 'background_q99')},
                'defect_coverage_in_shared_support': float(mask[common].sum() / mask.sum()) if mask.any() else None,
                'forward_seconds': {'baseline': seconds, 'crops': crop_seconds}}
            stem = '%03d_%s_%s' % (len(result['per_image']) + 1, source, row['class'])
            np.savez_compressed(out / 'cache' / (stem + '.npz'), baseline=base, baseline_margin=base_margin,
                crop_a=views[0], crop_b=views[1], margin_a=margins[0], margin_b=margins[1],
                shared_support=common, update_support=update_support, gt_mask=mask)
            row['cache'] = 'cache/' + stem + '.npz'
            result['per_image'].append(row)
            logger.info('%s anomaly=%d evidence_valid=%s soft_iou=%s contrast_min=%s AP_delta=%s background_delta=%s' % (
                stem, row['anomaly'], evidence['valid'], evidence['soft_iou'], evidence['contrast_min'],
                row['outcomes']['two_crop_ap_delta'], row['outcomes']['two_crop_background_delta']))
            output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    if len(result['per_image']) != len(manifest['validation']):
        raise RuntimeError('Validation identity count mismatch')
    result['unchanged_detector_tensors'] = frozen_guard(trainer, before, versions)
    result['by_source'] = {source: rank_analysis([r for r in result['per_image'] if r['source'] == source])
                           for source in meta['sources']}
    result['status'] = 'complete'
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    logger.info('Source reinspection reliability DONE; frozen detector PASS; report=' + str(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--reference_report', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--crop_fraction', type=float, default=.5)
    parser.add_argument('--shift_fraction', type=float, default=.05)
    parser.add_argument('--border_fraction', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=111)
    main(parser.parse_args())
