"""Source-only ALR feasibility probe; oracle crops are NOT deployable results.

Reuses the fixed validation identities in direct_gate_pilot.json. No training,
fusion, threshold selection, target evaluation, or checkpoint writes.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label
from sklearn.metrics import average_precision_score, roc_auc_score
import torch


MODES = ('baseline', 'raw_crop', 'resized_crop', 'global_prompt_crop')
BICUBIC = getattr(Image, 'Resampling', Image).BICUBIC
BILINEAR = getattr(Image, 'Resampling', Image).BILINEAR
CONFIG_KEYS = ('model', 'image_size', 'use_hsf', 'k_clusters', 'prompting_type',
               'prompting_branch', 'prompting_depth', 'prompting_length')


def resize_map(values, size):
    return np.asarray(Image.fromarray(np.asarray(values, dtype=np.float32), mode='F')
                      .resize(size, BILINEAR)).copy()


def choose_box(mask, baseline, fraction=0.5):
    """Largest GT component + context for anomalies; strongest window for normals.

    Coordinates are native image pixels, right/bottom exclusive. The GT choice
    is intentional and must never be presented as an automatic ALR selector.
    """
    h, w = mask.shape
    bw, bh = max(1, math.ceil(w * fraction)), max(1, math.ceil(h * fraction))
    components, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        ys, xs = np.where(components == sizes.argmax())
        bw = min(w, max(bw, math.ceil((xs.max() - xs.min() + 1) * 1.5)))
        bh = min(h, max(bh, math.ceil((ys.max() - ys.min() + 1) * 1.5)))
        cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
        x = min(max(round(cx - bw / 2), 0), w - bw)
        y = min(max(round(cy - bh / 2), 0), h - bh)
        policy = 'oracle_largest_gt_component_with_context'
    else:
        # Fixed 9x9 candidate grid, no pixel thresholds and no label tuning.
        summed = np.pad(baseline.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        candidates = []
        for y in np.unique(np.linspace(0, h - bh, 9).astype(int)):
            for x in np.unique(np.linspace(0, w - bw, 9).astype(int)):
                score = summed[y + bh, x + bw] - summed[y, x + bw] - summed[y + bh, x] + summed[y, x]
                candidates.append((float(score), int(x), int(y)))
        _, x, y = max(candidates, key=lambda item: item[0])
        policy = 'normal_highest_baseline_window_mean'
    return (int(x), int(y), int(x + bw), int(y + bh)), policy


def capture_prompts(model):
    result = []
    for branch in (model.visual_prompter, model.text_prompter):
        value = branch.dynamic_prompts
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise RuntimeError('Expected finite visual AND textual dynamic prompt tensors')
        result.append(value.detach().clone())
    return result


@contextmanager
def reuse_prompts(model, prompts):
    """Hold BOTH branches fixed without modifying the production forward path."""
    branches = (model.visual_prompter, model.text_prompter)
    old_prompts = [p.dynamic_prompts for p in branches]
    key = 'generate_and_set_dynamic_prompts'
    was_instance_attribute = key in model.__dict__
    previous = getattr(model, key)
    try:
        for branch, value in zip(branches, prompts):
            branch.set_dynamic_prompts(value)
        setattr(model, key, lambda *args, **kwargs: None)
        yield
    finally:
        if was_instance_attribute:
            setattr(model, key, previous)
        else:
            delattr(model, key)
        for branch, value in zip(branches, old_prompts):
            branch.set_dynamic_prompts(value)


def region_metrics(mask, prediction):
    if mask.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError('Invalid ROI prediction shape/values')
    target, score = mask.ravel().astype(bool), prediction.ravel()
    both = bool(target.any() and (~target).any())
    return {
        'pixel_ap': float(average_precision_score(target, score)) if both else None,
        'pixel_auroc': float(roc_auc_score(target, score)) if both else None,
        'foreground_mean': float(score[target].mean()) if target.any() else None,
        'background_mean': float(score[~target].mean()) if (~target).any() else None,
        'background_q99': float(np.quantile(score[~target], .99)) if (~target).any() else None,
    }


def summarize(rows):
    output = {'images': len(rows), 'mean_roi_metrics': {}, 'paired_vs_baseline': {}}
    for mode in MODES:
        output['mean_roi_metrics'][mode] = {}
        for key in rows[0]['metrics'][mode]:
            values = [r['metrics'][mode][key] for r in rows if r['metrics'][mode][key] is not None]
            output['mean_roi_metrics'][mode][key] = {'value': statistics.mean(values) if values else None,
                                                   'valid_images': len(values)}
        if mode == 'baseline':
            continue
        output['paired_vs_baseline'][mode] = {}
        for key in rows[0]['metrics'][mode]:
            deltas = [r['metrics'][mode][key] - r['metrics']['baseline'][key] for r in rows
                      if r['metrics'][mode][key] is not None and r['metrics']['baseline'][key] is not None]
            output['paired_vs_baseline'][mode][key] = {
                'mean_delta': statistics.mean(deltas) if deltas else None,
                'increases': sum(d > 1e-8 for d in deltas),
                'decreases': sum(d < -1e-8 for d in deltas),
                'ties': sum(abs(d) <= 1e-8 for d in deltas)}
    return output


def save_preview(path, original, mask, box, maps):
    """Same fixed 0..1 grayscale for ALL maps; no per-image min/max stretch."""
    size = 260
    canvas = Image.new('RGB', (size * 3, (size + 25) * 2), 'white')
    marked = original.copy()
    ImageDraw.Draw(marked).rectangle((box[0], box[1], box[2] - 1, box[3] - 1), outline='red',
                                    width=max(1, original.width // 200))
    tiles = [('Full image / red ROI', marked), ('GT within ROI', Image.fromarray(mask.astype('uint8') * 255))]
    tiles.extend((name + ' (ROI)', Image.fromarray((np.clip(maps[name], 0, 1) * 255).astype('uint8')))
                 for name in MODES)
    for i, (name, tile) in enumerate(tiles):
        x, y = (i % 3) * size, (i // 3) * (size + 25)
        canvas.paste(tile.convert('RGB').resize((size, size)), (x, y + 25))
        ImageDraw.Draw(canvas).text((x + 4, y + 5), name, fill='black')
    canvas.save(path)


def validate(args):
    baseline, reference = Path(args.baseline).resolve(), Path(args.reference_report).resolve()
    if not baseline.is_file() or not reference.is_file():
        raise FileNotFoundError('Baseline and original direct_gate_pilot.json must both exist')
    if not 0 < args.crop_fraction <= 1:
        raise ValueError('crop_fraction must be in (0, 1]')
    out = Path(args.save_path).resolve()
    if out in baseline.parents or out in reference.parents:
        raise ValueError('Use a separate output directory')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError('Use a new/empty save_path; existing results are preserved')
    report = json.loads(reference.read_text(encoding='utf8'))
    meta = report['metadata']
    if Path(meta['baseline_checkpoint']).resolve() != baseline:
        raise ValueError('Reference and supplied baseline checkpoint paths differ')
    if (meta['prompting_type'] != 'SD' or meta['prompting_branch'] != 'VL'
            or meta['sources'] != ['mvtec', 'colondb']):
        raise ValueError('This source probe requires the original SD/VL mvtec+colondb pilot')
    identities = set()
    for split in ('train', 'validation'):
        rows = report[split]['per_image']
        if not rows or len(rows) != report[split]['overall']['images']:
            raise ValueError('Reference split size mismatch')
        for row in rows:
            path = str(Path(row['image']).resolve())
            if (path in identities or row['anomaly'] not in (0, 1)
                    or row['source'] not in meta['sources']):
                raise ValueError('Duplicate/overlapping reference identity or invalid label/source')
            identities.add(path)
    return baseline, reference, out, report


def main(args):
    baseline, reference, out, reference_data = validate(args)
    # Delay CLIP imports until after validation, and keep geometry tests independent.
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
    (out / 'images').mkdir()
    logger = Logger(str(out / 'source_crop_pilot.txt'))
    logger.info('ORACLE SOURCE DIAGNOSTIC ONLY: no training, no target data, no formal ALR metrics.')
    model = AdaCLIP_Trainer(backbone=meta['model'],
        feat_list=[config['vision_cfg']['layers'] // 4 * i for i in (1, 2, 3, 4)],
        input_dim=config['vision_cfg']['width'], output_dim=config['embed_dim'], learning_rate=0.,
        device=device, image_size=meta['image_size'], prompting_depth=meta['prompting_depth'],
        prompting_length=meta['prompting_length'], prompting_branch='VL', prompting_type='SD',
        use_hsf=meta['use_hsf'], k_clusters=meta['k_clusters'], fusion_mode='add').to(device)
    before = import_baseline(model, str(baseline))
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    model.eval()
    versions = {k: v._version for k, v in model.clip_model.state_dict().items()}
    loaders, manifest = make_loaders(model, args, reference_data, get_data)
    logger.info('Strict baseline import PASS; validation images=%d' % len(manifest['validation']))
    result = {'metadata': {**{k: meta[k] for k in CONFIG_KEYS},
        'baseline_checkpoint': str(baseline), 'baseline_sha256': hashlib.sha256(baseline.read_bytes()).hexdigest(),
        'reference_report': str(reference), 'seed': args.seed, 'crop_fraction': args.crop_fraction,
        'sources': meta['sources'], 'source_validation_is_not_unseen_to_baseline': True,
        'metric_units': '0..1', 'evaluation': 'native-resolution ROI only; unfiltered softmax maps',
        'roi_policy': 'GT largest component plus context for anomalies; strongest window for normals',
        'normal_score_increases_are_not_thresholded_false_positive_counts': True,
        'fusion': 'none', 'global_prompt_control': 'both visual and text prompts held from full image',
        'resized_control': 'bicubic full-image resize then fractional extent crop, same native field of view',
        'no_target_evaluation': True}, 'status': 'running', 'per_image': []}
    report_path = out / 'source_crop_pilot.json'
    clip = model.clip_model

    def forward(tensor, cls_name, rng):
        _restore_rng(rng)
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            amap, score = clip(tensor.to(device), cls_name, aggregation=True)
        if device == 'cuda':
            torch.cuda.synchronize()
        if not torch.isfinite(amap).all() or not torch.isfinite(score).all():
            raise RuntimeError('Non-finite model outputs')
        return amap[0].float().cpu().numpy(), time.perf_counter() - start

    for source, loader in loaders['validation']:
        dataset = loader.dataset.dataset
        for index, items in zip(loader.dataset.indices, loader):
            record = dataset.data_all[index]
            path = Path(items['img_path'][0])
            with Image.open(path) as image:
                original = image.convert('RGB')
            w, h = original.size
            anomaly = int(items['anomaly'][0])
            mask = np.zeros((h, w), dtype=bool)
            if anomaly:
                with Image.open(Path(dataset.root) / record['mask_path']) as image:
                    mask = np.asarray(image.convert('L')) > 0
                if mask.shape != (h, w) or not mask.any():
                    raise ValueError('Anomalous image requires a nonempty native-size GT mask: ' + str(path))
            setup_seed(args.seed + len(result['per_image']))
            rng = _rng_state()
            base_small, base_seconds = forward(items['img'], items['cls_name'], rng)
            prompts = capture_prompts(clip)
            if not result['per_image']:
                with reuse_prompts(clip, prompts):
                    check, _ = forward(items['img'], items['cls_name'], rng)
                delta = float(np.abs(base_small - check).max())
                if not np.allclose(base_small, check, atol=1e-6, rtol=1e-5):
                    raise RuntimeError('Full-image prompt reuse equivalence failed')
                result['initial_equivalence'] = {'images': 1, 'max_abs_map_delta': delta, 'passed': True}
                logger.info('Full-image prompt reuse equivalence PASS; max map delta=%g' % delta)
            baseline_map = resize_map(base_small, (w, h))
            box, policy = choose_box(mask, baseline_map, args.crop_fraction)
            x0, y0, x1, y1 = box
            roi_mask = mask[y0:y1, x0:x1]
            raw = model.preprocess(original.crop(box)).unsqueeze(0)
            size = meta['image_size']
            lowres = original.resize((size, size), BICUBIC)
            # Identical native field of view, even for non-square original images.
            extent = (x0 * size / w, y0 * size / h, x1 * size / w, y1 * size / h)
            reduced_crop = lowres.transform((size, size), Image.EXTENT, extent, resample=BICUBIC)
            reduced = model.preprocess(reduced_crop).unsqueeze(0)
            maps = {'baseline': baseline_map[y0:y1, x0:x1]}
            times = {'baseline': base_seconds}
            for mode, tensor in (('raw_crop', raw), ('resized_crop', reduced), ('global_prompt_crop', raw)):
                if mode == 'global_prompt_crop':
                    with reuse_prompts(clip, prompts):
                        prediction, seconds = forward(tensor, items['cls_name'], rng)
                else:
                    prediction, seconds = forward(tensor, items['cls_name'], rng)
                maps[mode] = resize_map(prediction, (x1 - x0, y1 - y0))
                times[mode] = seconds
            row = {'source': source, 'class': items['cls_name'][0], 'image': str(path), 'anomaly': anomaly,
                   'native_size': [w, h], 'box_xyxy': list(box), 'selection': policy,
                   'defect_area_fraction': float(mask.mean()),
                   'defect_coverage': float(roi_mask.sum() / mask.sum()) if anomaly else None,
                   'crop_area_fraction': (x1 - x0) * (y1 - y0) / (w * h),
                   'metrics': {mode: region_metrics(roi_mask, maps[mode]) for mode in MODES},
                   'forward_seconds': times}
            result['per_image'].append(row)
            stem = '%03d_%s_%s' % (len(result['per_image']), source, row['class'])
            save_preview(out / 'images' / (stem + '.png'), original, roi_mask, box, maps)
            logger.info('%s anomaly=%d AP=%s background_mean=%s' % (stem, anomaly,
                {m: row['metrics'][m]['pixel_ap'] for m in MODES},
                {m: row['metrics'][m]['background_mean'] for m in MODES}))
            report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    if len(result['per_image']) != len(manifest['validation']):
        raise RuntimeError('Validation identity count mismatch')
    result['unchanged_detector_tensors'] = frozen_guard(model, before, versions)
    result['overall'] = summarize(result['per_image'])
    result['by_source_label'] = {}
    for source in meta['sources']:
        for anomaly in (0, 1):
            rows = [r for r in result['per_image'] if r['source'] == source and r['anomaly'] == anomaly]
            if rows:
                result['by_source_label'][source + ':' + str(anomaly)] = summarize(rows)
    result['status'] = 'complete'
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    logger.info('Source crop pilot DONE; detector unchanged; diagnostic ROI metrics only; report=' + str(report_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--reference_report', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--crop_fraction', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=111)
    main(parser.parse_args())
