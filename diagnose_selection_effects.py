"""Offline selection diagnosis. No torch, model inference or rule fitting.

Compare cached baseline/adaptive/fixed-center maps at the image and class levels.
A fixed, label-free per-image empirical CDF preserves within-image ranks/ties,
but changes cross-image ranking. Its pooled AP is a diagnostic counterfactual,
NOT a proposed deployment normalization or an additive AP decomposition.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from audit_reinspection_risk import safe_child
from diagnose_reinspection_ranking import image_ap, mean


MODES = ('baseline', 'adaptive', 'fixed_center')
PAIRS = (('adaptive', 'baseline'), ('adaptive', 'fixed_center'), ('fixed_center', 'baseline'))
GAIN_EPS = 1e-4  # 0.01 percentage point.


def percentile_map(values):
    """Average ranks preserve ties. Constant maps become exactly 0.5."""
    if values.size > 2 ** 23:
        raise ValueError('Float32 rank cache requires at most 2**23 pixels per image')
    return ((rankdata(values.ravel(), method='average') - .5) / values.size).astype(np.float32).reshape(values.shape)


def map_statistics(mask, values):
    # image_ap validates the mask, shape, finite values and probability range.
    ap = image_ap(mask, values)
    foreground, background = values[mask], values[~mask]
    return dict(ap=ap, foreground_mean=float(foreground.mean()) if foreground.size else None,
        background_mean=float(background.mean()) if background.size else None,
        background_q99=float(np.quantile(background, .99)) if background.size else None,
        mean=float(values.mean()), q99=float(np.quantile(values, .99)))


def changes(values, epsilon=GAIN_EPS):
    return dict(defined_images=len(values), mean_delta=mean(values),
        median_delta=float(np.median(values)) if values else None,
        increases=sum(v > epsilon for v in values), decreases=sum(v < -epsilon for v in values),
        near_unchanged=sum(abs(v) <= epsilon for v in values))


def compare_images(rows, left, right):
    def deltas(group, metric):
        return [r['metrics'][left][metric] - r['metrics'][right][metric] for r in group
                if r['metrics'][left][metric] is not None and r['metrics'][right][metric] is not None]
    annotated = [r for r in rows if r['anomaly'] == 1]
    normal = [r for r in rows if r['anomaly'] == 0]
    valid_both = [r for r in annotated if r['geometry_valid'][left] and r['geometry_valid'][right]]
    return dict(anomalous_image_ap=changes(deltas(annotated, 'ap')),
        anomalous_image_ap_when_both_geometrically_valid=changes(deltas(valid_both, 'ap')),
        anomalous_background_q99=changes(deltas(annotated, 'background_q99'), 1e-12),
        normal_mean_response=changes(deltas(normal, 'mean'), 1e-12),
        normal_q99_response=changes(deltas(normal, 'q99'), 1e-12))


def validate_report(path):
    report = json.loads(path.read_text(encoding='utf8'))
    protocol = report['protocol']
    if report.get('status') != 'complete' or report.get('unchanged_detector_tensors', 0) <= 0:
        raise ValueError('Require a completed target selection report with frozen-detector PASS')
    if not report.get('initial_equivalence', {}).get('passed'):
        raise ValueError('Initial held-prompt equivalence must have passed')
    if (any(protocol.get(k) is not True for k in ('no_training', 'no_score_gate', 'no_target_threshold_tuning'))
            or protocol.get('crops_per_variant') != 2 or protocol.get('fusion_weight') != .5):
        raise ValueError('Expected the fixed two-crop/no-gate selection protocol')
    name = report['testing_data']
    if not isinstance(name, str) or not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name):
        raise ValueError('Unsafe dataset name')
    rows = report['per_image']
    identities = [str(Path(r['image']).resolve()) for r in rows]
    classes = list(report['metrics'])
    if not rows or len(set(identities)) != len(rows) or set(classes) != {r['class'] for r in rows}:
        raise ValueError('Invalid class/image identities')
    guard = report.get('reference_equivalence', {})
    if guard.get('enabled') and (guard.get('passed') is not True or guard.get('checked_images') != len(rows)):
        raise ValueError('Enabled reference equivalence did not pass for every image')
    side = protocol['source_config']['image_size']
    if not isinstance(side, int) or side < 1 or side * side > 2 ** 23:
        raise ValueError('Invalid image size for rank-preserving float32 caches')
    for i, category in enumerate(classes):
        group = [r for r in rows if r['class'] == category]
        if sorted(r['cache_position'] for r in group) != list(range(len(group))):
            raise ValueError('Invalid cache positions: ' + category)
        if any(r['anomaly'] not in (0, 1) for r in group):
            raise ValueError('Invalid anomaly label')
        for mode in MODES + ('mask',):
            array = np.load(safe_child(path.parent, 'cache/class_%02d_%s.npy' % (i, mode)),
                            mmap_mode='r', allow_pickle=False)
            if array.shape != (len(group), side, side) or (mode == 'mask' and array.dtype != np.bool_):
                raise ValueError('Invalid cache shape/dtype: ' + category + '/' + mode)
        for mode in MODES:
            value = report['metrics'][category][mode]['ap_px']
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError('Invalid reported pooled AP')
    for mode in MODES:
        expected = mean([report['metrics'][c][mode]['ap_px'] for c in classes])
        if not math.isclose(expected, report['average'][mode]['ap_px'], abs_tol=1e-6, rel_tol=0):
            raise ValueError('Reported category average AP mismatch')
    return report


def pooled_ap(mask, values):
    # Match the existing official helper's degenerate-GT convention.
    if not mask.any() or mask.all():
        return 0.
    return float(average_precision_score(mask.ravel(), values.ravel())) * 100


def diagnose_class(path, report, class_index, category, scratch):
    group = [r for r in report['per_image'] if r['class'] == category]
    cache = {mode: np.load(safe_child(path.parent, 'cache/class_%02d_%s.npy' % (class_index, mode)),
                          mmap_mode='r', allow_pickle=False) for mode in MODES + ('mask',)}
    # Rank maps are temporary, one class at a time. Original files stay read-only.
    ranks = {mode: np.lib.format.open_memmap(scratch / (mode + '.npy'), mode='w+',
              dtype=np.float32, shape=cache['mask'].shape) for mode in MODES}
    records = []
    for j, original in enumerate(group):
        position = original['cache_position']
        mask = cache['mask'][position]
        if original['anomaly'] == 0 and mask.any():
            raise ValueError('Normal image has positive GT pixels')
        statistics = {}
        for mode in MODES:
            values = cache[mode][position]
            statistics[mode] = map_statistics(mask, values)
            ranks[mode][position] = percentile_map(values)
        coverage = {mode: original['defect_coverage'][mode] for mode in ('adaptive', 'fixed_center')}
        records.append(dict(image=original['image'], **{'class': category}, anomaly=original['anomaly'],
            cache_position=position, foreground_pixels=int(mask.sum()), metrics=statistics,
            geometry_valid={'baseline': True, **{m: original['selections'][m]['geometry_valid'] for m in MODES[1:]}},
            defect_coverage=coverage,
            update_fraction={m: original['selections'][m]['update_fraction'] for m in MODES[1:]}))
        if (j + 1) % 32 == 0:
            print('%s class=%s per-image diagnostics=%d/%d' % (report['testing_data'], category, j + 1, len(group)), flush=True)
    raw, normalized = {}, {}
    for mode in MODES:
        print('%s class=%s pooled AP replay/CDF=%s' % (report['testing_data'], category, mode), flush=True)
        raw[mode] = pooled_ap(cache['mask'], cache[mode])
        if not math.isclose(raw[mode], report['metrics'][category][mode]['ap_px'], abs_tol=1e-6, rel_tol=0):
            raise ValueError('Cached/reported pooled AP mismatch: %s/%s' % (category, mode))
        ranks[mode].flush()
        normalized[mode] = pooled_ap(cache['mask'], ranks[mode])
    diagnostic = dict(images=len(group), anomalous_images=sum(r['anomaly'] == 1 for r in group),
        normal_images=sum(r['anomaly'] == 0 for r in group),
        anomalous_images_without_defined_ap=sum(r['anomaly'] == 1 and r['metrics']['baseline']['ap'] is None for r in records),
        single_image_ap_mean={m: mean([r['metrics'][m]['ap'] for r in records if r['metrics'][m]['ap'] is not None]) for m in MODES},
        pooled_ap_percent=raw, rank_normalized_pooled_ap_percent=normalized, comparisons={})
    for left, right in PAIRS:
        pair = compare_images(records, left, right)
        raw_delta, rank_delta = raw[left] - raw[right], normalized[left] - normalized[right]
        image_delta = pair['anomalous_image_ap']['mean_delta']
        pair.update(pooled_ap_delta_pp=raw_delta, rank_normalized_pooled_ap_delta_pp=rank_delta,
            single_image_gain_but_pooled_loss=image_delta is not None and image_delta > GAIN_EPS and raw_delta < -.01,
            single_image_loss_but_pooled_gain=image_delta is not None and image_delta < -GAIN_EPS and raw_delta > .01,
            pooled_loss_becomes_gain_under_cdf=raw_delta < -.01 and rank_delta > .01)
        diagnostic['comparisons'][left + '_vs_' + right] = pair
    del ranks
    return diagnostic, records


def dataset_summary(report, classes, rows):
    result = dict(images=len(rows), by_class=classes,
        original_dataset_metrics={m: report['average'][m] for m in MODES},
        class_balanced_single_image_ap={m: mean([d['single_image_ap_mean'][m] for d in classes.values()
                                                if d['single_image_ap_mean'][m] is not None]) for m in MODES},
        rank_normalized_pooled_ap_percent={m: mean([d['rank_normalized_pooled_ap_percent'][m] for d in classes.values()]) for m in MODES},
        comparisons={})
    for left, right in PAIRS:
        key = left + '_vs_' + right
        pair = compare_images(rows, left, right)
        a, b = (result['class_balanced_single_image_ap'][m] for m in (left, right))
        pair.update(class_balanced_single_image_ap_delta=a - b if a is not None and b is not None else None,
            pooled_ap_delta_pp=report['average'][left]['ap_px'] - report['average'][right]['ap_px'],
            rank_normalized_pooled_ap_delta_pp=result['rank_normalized_pooled_ap_percent'][left] - result['rank_normalized_pooled_ap_percent'][right],
            classes_with_single_image_gain_but_pooled_loss=[c for c, d in classes.items() if d['comparisons'][key]['single_image_gain_but_pooled_loss']],
            classes_with_cdf_reversing_pooled_loss=[c for c, d in classes.items() if d['comparisons'][key]['pooled_loss_becomes_gain_under_cdf']])
        result['comparisons'][key] = pair
    defined = [r for r in rows if r['metrics']['adaptive']['ap'] is not None]
    delta = lambda r: r['metrics']['adaptive']['ap'] - r['metrics']['fixed_center']['ap']
    result['examples'] = dict(
        adaptive_worse_than_center=sorted([r for r in defined if delta(r) < -GAIN_EPS], key=delta)[:8],
        adaptive_better_than_center=sorted([r for r in defined if delta(r) > GAIN_EPS], key=delta, reverse=True)[:8],
        higher_defect_coverage_but_lower_ap=sorted([r for r in defined if delta(r) < -GAIN_EPS
            and r['defect_coverage']['adaptive'] is not None and r['defect_coverage']['fixed_center'] is not None
            and r['defect_coverage']['adaptive'] > r['defect_coverage']['fixed_center']], key=delta)[:8])
    return result


def main(args):
    paths = [Path(p).resolve() for p in args.eval_reports]
    output = Path(args.save_path).resolve()
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('Supply distinct target selection reports')
    if any(output in p.parents or p.parent in output.parents for p in paths):
        raise ValueError('Use output separate from input experiment directories')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Use a new/empty diagnosis directory')
    reports = [validate_report(p) for p in paths]
    if len({r['testing_data'] for r in reports}) != len(reports):
        raise ValueError('One report per dataset is required')
    identities = [str(Path(row['image']).resolve()) for report in reports for row in report['per_image']]
    if len(identities) != len(set(identities)):
        raise ValueError('Image identities overlap between target reports')
    for report in reports[1:]:
        if report['protocol']['source_config'] != reports[0]['protocol']['source_config']:
            raise ValueError('Source configurations differ between target reports')
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(status='running', no_model_inference=True, deployment_rules_changed=False,
        ap_gain_epsilon=GAIN_EPS, units='single-image AP: 0..1; pooled AP and pooled deltas: percentage points',
        protocol=dict(modes=list(MODES), no_second_smoothing=True, all_images_in_pooled_ap=True,
            normal_images_have_no_single_image_ap=True,
            cdf='(average_rank(score)-0.5)/pixel_count within each image; ties preserved; label-free and fixed',
            cdf_is_diagnostic_not_a_deployment_rule=True, no_hyperparameter_search=True,
            caveats='Per-image and pooled AP differ in pixel weighting and cross-image ordering. '
                    'Their disagreement alone does not prove calibration failure. CDF changes only '
                    'cross-image score ordering, but can remove useful confidence; it is not a causal '
                    'attribution of model failure. Normal response changes are NOT false-positive rates.'), datasets={})
    summary_path = output / 'selection_effects_summary.json'
    for path, report in zip(paths, reports):
        classes, rows = {}, []
        for i, category in enumerate(report['metrics']):
            with tempfile.TemporaryDirectory(prefix='rank_scratch_', dir=output) as temporary:
                diagnostic, records = diagnose_class(path, report, i, category, Path(temporary))
            classes[category] = diagnostic
            rows.extend(records)
        name = report['testing_data']
        summary['datasets'][name] = dataset_summary(report, classes, rows)
        summary['datasets'][name]['input_report_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        (output / (name + '_per_image_effects.json')).write_text(json.dumps(rows, indent=2, allow_nan=False), encoding='utf8')
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf8')
    summary['status'] = 'complete'
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf8')
    lines = ['Selection effects diagnosis DONE', 'No model inference or target deployment-rule fitting.',
             'CDF pooled AP is a diagnostic counterfactual, NOT a deployment result.']
    for name, dataset in summary['datasets'].items():
        lines.append('%s images=%d' % (name, dataset['images']))
        for key, comparison in dataset['comparisons'].items():
            lines.append('  %s single_image_macro_delta=%s pooled_delta_pp=%.6f CDF_pooled_delta_pp=%.6f' % (
                key, comparison['class_balanced_single_image_ap_delta'], comparison['pooled_ap_delta_pp'],
                comparison['rank_normalized_pooled_ap_delta_pp']))
        for category, d in dataset['by_class'].items():
            c = d['comparisons']['adaptive_vs_fixed_center']
            lines.append('  %s adaptive_vs_center image_AP_delta=%s pooled_delta_pp=%.6f CDF_delta_pp=%.6f normal_q99_delta=%s' % (
                category, c['anomalous_image_ap']['mean_delta'], c['pooled_ap_delta_pp'],
                c['rank_normalized_pooled_ap_delta_pp'], c['normal_q99_response']['mean_delta']))
    text = '\n'.join(lines) + '\n'
    (output / 'selection_effects_summary.txt').write_text(text, encoding='utf8')
    print(text, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval_reports', nargs='+', required=True)
    parser.add_argument('--save_path', required=True)
    main(parser.parse_args())
