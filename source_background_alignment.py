"""Offline source-only background-shift ablation. No torch/model inference.

Within the existing update support, baseline scores <= their median define
prediction-only anchors (all cutoff ties retained). Subtract the SIGNED median
of (mean local prediction - baseline) on anchors, clip the local prediction,
then use the original 0.5 fusion. No per-image or per-class parameter search.
GT is used only AFTER candidate construction for evaluation/anchor diagnostics.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score

from audit_reinspection_risk import (safe_child, calibrate_pixel_threshold,
                                    pixel_counts, empty_counts, accumulate, finish)
from calibrate_reinspection import CONFIG_KEYS, validate_report
from diagnose_reinspection_ranking import mean
from diagnose_selection_effects import pooled_ap


MODES = ('baseline', 'ordinary_fusion', 'background_aligned')
REPLAY_METRICS = ('pixel_ap', 'pixel_auroc', 'background_mean', 'background_q99')
AP_EPS = 1e-4


def candidates(base, a, b, support):
    """Prediction-only: intentionally no GT, label, class or source arguments."""
    if base.ndim != 2 or support.shape != base.shape or support.dtype != np.bool_:
        raise ValueError('Invalid update support shape/dtype')
    for value in (base, a, b):
        if (value.shape != base.shape or not np.issubdtype(value.dtype, np.floating)
                or not np.isfinite(value).all() or value.min() < 0 or value.max() > 1):
            raise ValueError('Invalid raw probability map')
    maps = {mode: base.copy() for mode in MODES}
    anchors = np.zeros_like(support)
    info = dict(update_pixels=int(support.sum()), anchor_pixels=0, anchor_fraction_of_support=None,
                anchor_cutoff=None, signed_offset=None, clipped_low_fraction=None,
                clipped_high_fraction=None, mean_local_change_in_support=None)
    if not support.any():
        return maps, anchors, info
    local = (a + b) / 2
    cutoff = float(np.median(base[support]))
    anchors = support & (base <= cutoff)
    offset = float(np.median((local - base)[anchors]))
    unbounded = local[support] - offset
    aligned = np.clip(unbounded, 0, 1)
    maps['ordinary_fusion'][support] = .5 * base[support] + .5 * local[support]
    maps['background_aligned'][support] = .5 * base[support] + .5 * aligned
    info.update(anchor_pixels=int(anchors.sum()), anchor_fraction_of_support=float(anchors.sum() / support.sum()),
                anchor_cutoff=cutoff, signed_offset=offset,
                clipped_low_fraction=float(np.mean(unbounded < 0)),
                clipped_high_fraction=float(np.mean(unbounded > 1)),
                mean_local_change_in_support=float((aligned - local[support]).mean()))
    return maps, anchors, info


def metrics(mask, values):
    foreground, background = values[mask], values[~mask]
    both = bool(mask.any() and not mask.all())
    return dict(pixel_ap=float(average_precision_score(mask.ravel(), values.ravel())) if both else None,
        pixel_auroc=float(roc_auc_score(mask.ravel(), values.ravel())) if both else None,
        foreground_mean=float(foreground.mean()) if foreground.size else None,
        background_mean=float(background.mean()) if background.size else None,
        background_q99=float(np.quantile(background, .99)) if background.size else None)


def prepare(args):
    paths = [Path(args.calibration_report).resolve(), Path(args.validation_report).resolve()]
    output = Path(args.save_path).resolve()
    if paths[0] == paths[1] or any(output in p.parents or p.parent in output.parents for p in paths):
        raise ValueError('Use distinct source reports and a separate output directory')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Use a new/empty output directory')
    reports = [json.loads(p.read_text(encoding='utf8')) for p in paths]
    if reports[0]['metadata'].get('split') != 'train':
        raise ValueError('Calibration report must be explicitly marked split=train')
    # Older validation reports omit split; exact original reference identities
    # below still prove that they are the validation subset, not target data.
    if reports[1]['metadata'].get('split', 'validation') != 'validation':
        raise ValueError('Validation report must use the source validation subset')
    reference_path = Path(reports[0]['metadata']['reference_report']).resolve()
    if output in reference_path.parents or reference_path.parent in output.parents:
        raise ValueError('Output must be separate from original source reference')
    reference = json.loads(reference_path.read_text(encoding='utf8'))
    for key in CONFIG_KEYS:
        if any(key not in report['metadata'] for report in reports):
            raise ValueError('Missing source configuration field: ' + key)
        if reports[0]['metadata'].get(key) != reports[1]['metadata'].get(key):
            raise ValueError('Source split configuration mismatch: ' + key)
    for path, report, split in zip(paths, reports, ('train', 'validation')):
        meta = report['metadata']
        if (meta.get('sources') != ['mvtec', 'colondb'] or meta.get('no_target_evaluation') is not True
                or meta.get('candidate_fusion_weight') != .5
                or Path(meta['reference_report']).resolve() != reference_path):
            raise ValueError('Only the original fixed-fusion mvtec+colondb source reports are allowed')
        if not report.get('initial_equivalence', {}).get('passed'):
            raise ValueError('Source held-prompt equivalence must have passed')
        if not isinstance(meta['image_size'], int) or meta['image_size'] < 1:
            raise ValueError('Invalid source image size')
        validate_report(report, reference[split]['per_image'], split)
        for row in report['per_image']:
            if row['source'] not in meta['sources']:
                raise ValueError('Unexpected source dataset')
            safe_child(path.parent, row['cache'])
    identities = [str(Path(r['image']).resolve()) for report in reports for r in report['per_image']]
    if len(identities) != len(set(identities)):
        raise ValueError('Source subsets overlap')
    return paths, reports, output, reference_path


def load_cache(root, row, side):
    with np.load(safe_child(root, row['cache']), allow_pickle=False) as cache:
        arrays = {k: cache[k] for k in ('baseline', 'crop_a', 'crop_b', 'shared_support', 'update_support', 'gt_mask')}
    if any(v.shape != (side, side) for v in arrays.values()):
        raise ValueError('Raw source cache shape mismatch')
    if any(arrays[k].dtype != np.bool_ for k in ('shared_support', 'update_support', 'gt_mask')):
        raise ValueError('Source masks/supports must be bool')
    expected = arrays['shared_support'] if row['evidence']['valid'] else np.zeros((side, side), bool)
    if not np.array_equal(expected, arrays['update_support']):
        raise ValueError('Cached support differs from the original geometry policy')
    return arrays


def check_replay(recorded, replayed):
    maximum = 0.
    for key in REPLAY_METRICS:
        before, after = recorded[key], replayed[key]
        if ((before is None) != (after is None)
                or (before is not None and not math.isclose(before, after, abs_tol=1e-8, rel_tol=1e-6))):
            raise ValueError('Raw cache/recorded metric replay mismatch: ' + key)
        if before is not None:
            maximum = max(maximum, abs(before - after))
    return maximum


def summarize(rows, class_ap):
    categories = sorted({r['class'] for r in rows})
    result = dict(images=len(rows), normal_images=sum(r['anomaly'] == 0 for r in rows),
                  anomalous_images=sum(r['anomaly'] == 1 for r in rows), modes={}, comparisons={})
    for mode in MODES:
        entry = dict(class_balanced={}, image_average={}, operating_point=empty_counts(),
                     selected_subset_class_pooled_ap_percent=mean([class_ap[c][mode] for c in categories]))
        for metric in ('pixel_ap', 'pixel_auroc', 'foreground_mean', 'background_mean', 'background_q99'):
            label = 0 if metric.startswith('background') else 1
            selected = [r for r in rows if r['anomaly'] == label and r['metrics'][mode][metric] is not None]
            entry['image_average'][metric] = mean([r['metrics'][mode][metric] for r in selected])
            means = [mean([r['metrics'][mode][metric] for r in selected if r['class'] == c]) for c in categories]
            entry['class_balanced'][metric] = mean([v for v in means if v is not None])
        for row in rows:
            accumulate(entry['operating_point'], row['pixel_counts'][mode], row['anomaly'] == 0)
        entry['operating_point'] = finish(entry['operating_point'])
        result['modes'][mode] = entry
    for right in ('ordinary_fusion', 'baseline'):
        left = 'background_aligned'
        deltas = [r['metrics'][left]['pixel_ap'] - r['metrics'][right]['pixel_ap'] for r in rows
                  if r['anomaly'] == 1 and r['metrics'][left]['pixel_ap'] is not None]
        a, b = result['modes'][left], result['modes'][right]
        def difference(aa, bb):
            return aa - bb if aa is not None and bb is not None else None
        result['comparisons'][left + '_vs_' + right] = dict(
            class_balanced_anomalous_ap_delta=difference(a['class_balanced']['pixel_ap'], b['class_balanced']['pixel_ap']),
            selected_subset_pooled_ap_delta_pp=difference(a['selected_subset_class_pooled_ap_percent'], b['selected_subset_class_pooled_ap_percent']),
            normal_background_mean_delta=difference(a['class_balanced']['background_mean'], b['class_balanced']['background_mean']),
            normal_background_q99_delta=difference(a['class_balanced']['background_q99'], b['class_balanced']['background_q99']),
            normal_pixel_fpr_delta=difference(a['operating_point']['normal_pixel_fpr'], b['operating_point']['normal_pixel_fpr']),
            foreground_recall_delta=difference(a['operating_point']['pixel_recall'], b['operating_point']['pixel_recall']),
            defined_ap_images=len(deltas), improved_ap_images=sum(d > AP_EPS for d in deltas),
            worsened_ap_images=sum(d < -AP_EPS for d in deltas), near_unchanged_ap_images=sum(abs(d) <= AP_EPS for d in deltas))
    active = [r for r in rows if r['alignment']['signed_offset'] is not None]
    offsets = [r['alignment']['signed_offset'] for r in active]
    result['alignment_diagnostics'] = dict(valid_update_images=len(active),
        positive_offsets=sum(v > 1e-12 for v in offsets), negative_offsets=sum(v < -1e-12 for v in offsets),
        near_zero_offsets=sum(abs(v) <= 1e-12 for v in offsets), median_signed_offset=float(np.median(offsets)) if offsets else None,
        mean_anchor_gt_foreground_fraction=mean([r['anchor_gt_foreground_fraction'] for r in active]),
        mean_clip_low_fraction=mean([r['alignment']['clipped_low_fraction'] for r in active]),
        mean_clip_high_fraction=mean([r['alignment']['clipped_high_fraction'] for r in active]))
    result['alignment_diagnostics']['anomalous_anchor_gt_foreground_fraction'] = mean([
        r['anchor_gt_foreground_fraction'] for r in active if r['anomaly'] == 1])
    result['alignment_diagnostics']['anomalous_images_with_gt_in_anchors'] = sum(
        r['anomaly'] == 1 and r['anchor_gt_foreground_fraction'] > 0 for r in active)
    return result


def diagnose_split(path, report, output, threshold):
    rows, class_ap, replay_max = [], {}, 0.
    side = report['metadata']['image_size']
    groups = sorted({(r['source'], r['class']) for r in report['per_image']})
    for source, category in groups:
        originals = [r for r in report['per_image'] if r['source'] == source and r['class'] == category]
        with tempfile.TemporaryDirectory(prefix='source_alignment_', dir=output) as temporary:
            scratch = Path(temporary)
            pools = {m: np.lib.format.open_memmap(scratch / (m + '.npy'), mode='w+',
                dtype=np.float32, shape=(len(originals), side, side)) for m in MODES}
            # Boolean GT is kept separate from the float32 prediction maps.
            masks = np.lib.format.open_memmap(scratch / 'mask_bool.npy', mode='w+',
                                              dtype=np.bool_, shape=(len(originals), side, side))
            for i, original in enumerate(originals):
                cache = load_cache(path.parent, original, side)
                updates, anchors, info = candidates(cache['baseline'], cache['crop_a'], cache['crop_b'], cache['update_support'])
                # All candidate decisions are now fixed. GT enters ONLY below.
                mask = cache['gt_mask']
                if original['anomaly'] == 0 and mask.any():
                    raise ValueError('Normal source image contains positive GT pixels')
                filtered = {m: gaussian_filter(v, sigma=4) for m, v in updates.items()}
                calculated = {m: metrics(mask, v) for m, v in filtered.items()}
                replay_max = max(replay_max, check_replay(original['metrics']['baseline'], calculated['baseline']),
                    check_replay(original['metrics']['two_crop_fixed_fusion'], calculated['ordinary_fusion']))
                record = dict(source=source, **{'class': category}, image=original['image'], anomaly=original['anomaly'],
                    original_cache=original['cache'], alignment=info, metrics=calculated,
                    pixel_counts={m: pixel_counts(mask, v, threshold) for m, v in filtered.items()},
                    anchor_gt_foreground_fraction=float(mask[anchors].mean()) if anchors.any() else None)
                rows.append(record)
                for m in MODES:
                    pools[m][i] = filtered[m]
                masks[i] = mask
            for m in MODES:
                pools[m].flush()
            class_ap[(source, category)] = {m: pooled_ap(masks, pools[m]) for m in MODES}
            del pools, masks
        print('%s %s/%s images=%d replay_max_delta=%g' % (
            report['metadata'].get('split', 'validation'), source, category, len(originals), replay_max), flush=True)
    result = dict(images=len(rows), replay_passed=True, replay_max_absolute_metric_delta=replay_max,
                  by_source={}, by_class={})
    for source in report['metadata']['sources']:
        selected = [r for r in rows if r['source'] == source]
        ap = {category: value for (dataset, category), value in class_ap.items() if dataset == source}
        result['by_source'][source] = summarize(selected, ap)
        result['by_class'][source] = {c: summarize([r for r in selected if r['class'] == c], {c: ap[c]}) for c in ap}
    return result, rows


def main(args):
    paths, reports, output, reference = prepare(args)
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(no_model_inference=True, no_training=True, no_target_data=True,
        no_parameter_search=True, no_validation_rule_selection=True, fusion_weight=.5,
        modes=list(MODES), reference_quantile=.5, include_all_cutoff_ties=True,
        anchors='update_support AND baseline <= median(baseline[update_support]); prediction-only',
        signed_offset='median((crop_a+crop_b)/2 - baseline) on anchors',
        correction='clip(mean_local_prediction - signed_offset,0,1) inside original update_support only',
        smoothing='sigma4 exactly once AFTER fusion, same as original source diagnostic',
        positive_offset_lowers_local_scores=True, negative_offset_raises_local_scores=True,
        operating_threshold_fitted_on_source_train_normals_only=True,
        operating_threshold_does_not_control_alignment=True,
        reused_source_validation_is_exploratory=True,
        raw_maps_outside_support_unchanged=True, smoothed_maps_can_change_near_support_boundaries=True,
        anchor_gt_contamination_is_evaluation_only=True,
        caveats='Low initial response does not guarantee true background. Signed alignment can raise '
                'normal responses or remove useful anomaly evidence. This is a fixed source-only '
                'ablation, not a guarantee of reliability or an unseen benchmark result.',
        metric_units='per-image AP/AUROC/responses/FPR/recall: 0..1; selected-subset pooled AP: 0..100',
        source_reports={label: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                        for label, path in zip(('train', 'validation'), paths)},
        reference_report_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        source_config={k: reports[0]['metadata'][k] for k in CONFIG_KEYS})
    (output / 'fixed_background_alignment_protocol.json').write_text(json.dumps(protocol, indent=2, allow_nan=False), encoding='utf8')
    print('Calibrating one fixed operating point from SOURCE train-normal baseline pixels only...', flush=True)
    operating = calibrate_pixel_threshold(reports[0], paths[0].parent, output, source_pixel_fpr=.001)
    summary = dict(status='running', protocol=protocol, operating_point=operating, splits={})
    summary_path = output / 'source_background_alignment_summary.json'
    for label, path, report in zip(('train', 'validation'), paths, reports):
        result, rows = diagnose_split(path, report, output, operating['pixel_threshold'])
        summary['splits'][label] = result
        (output / (label + '_per_image_alignment.json')).write_text(json.dumps(rows, indent=2, allow_nan=False), encoding='utf8')
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf8')
    summary['status'] = 'complete'
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf8')
    lines = ['Source background alignment DONE', 'No inference, training or target/validation rule selection.',
             'Signed median shift; negative offsets RAISE local scores. Fixed source-normal operating point.',
             'Per-image AP and selected-subset category-pooled AP are different metrics.']
    for split, d in summary['splits'].items():
        lines.append('%s images=%d raw-cache replay PASS max_delta=%g' % (split, d['images'], d['replay_max_absolute_metric_delta']))
        for source, values in d['by_source'].items():
            for mode, v in values['modes'].items():
                lines.append('  %s %s macro_image_AP=%s subset_pooled_AP=%s normal_pixel_FPR=%s foreground_recall=%s normal_mean=%s normal_q99=%s' % (
                    source, mode, v['class_balanced']['pixel_ap'], v['selected_subset_class_pooled_ap_percent'],
                    v['operating_point']['normal_pixel_fpr'], v['operating_point']['pixel_recall'],
                    v['class_balanced']['background_mean'], v['class_balanced']['background_q99']))
            lines.append('  %s comparisons=%s' % (source, json.dumps(values['comparisons'], allow_nan=False)))
            lines.append('  %s offsets=%s' % (source, json.dumps(values['alignment_diagnostics'], allow_nan=False)))
    text = '\n'.join(lines) + '\n'
    (output / 'source_background_alignment_summary.txt').write_text(text, encoding='utf8')
    print(text, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration_report', required=True)
    parser.add_argument('--validation_report', required=True)
    parser.add_argument('--save_path', required=True)
    main(parser.parse_args())
