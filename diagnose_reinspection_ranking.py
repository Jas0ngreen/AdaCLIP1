"""Offline utility ranking diagnosis for completed target evaluation caches.

No torch, model calls, retraining, or target-based deployment threshold selection.
Predeclared coverage points are descriptive ranking diagnostics, not fitted rules.
Per-image AP is NOT the category-pooled AP used in the full evaluation CSVs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from audit_reinspection_risk import safe_child, validate_inputs


FEATURES = ('soft_iou', 'baseline_core_mean', 'global_image_score')
COVERAGES = (.1, .25, .5, .75, 1.)
GAIN_EPS = 1e-4


def mean(values):
    return float(np.mean(values)) if values else None


def correlation(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = float(spearmanr(x, y).correlation)
    return value if math.isfinite(value) else None


def image_ap(mask, values):
    if mask.dtype != np.bool_ or values.shape != mask.shape or not np.isfinite(values).all():
        raise ValueError('Invalid cached mask/prediction')
    if np.min(values) < 0 or np.max(values) > 1:
        raise ValueError('Invalid cached probability range')
    if not mask.any() or mask.all():
        return None
    return float(average_precision_score(mask.ravel(), values.ravel()))


def associations(rows, feature):
    eligible = [r for r in rows if r['valid'] and r['ap_gain'] is not None]
    x = [r['scores'][feature] for r in eligible]
    y = [r['ap_gain'] for r in eligible]
    useful = np.asarray(y) > GAIN_EPS
    decisive = [r for r in eligible if abs(r['ap_gain']) > GAIN_EPS]
    truth = np.asarray([r['ap_gain'] > GAIN_EPS for r in decisive])
    return {'valid_anomalous_images': len(eligible),
            'beneficial': int(np.count_nonzero(useful)),
            'harmful': sum(r['ap_gain'] < -GAIN_EPS for r in eligible),
            'near_unchanged': sum(abs(r['ap_gain']) <= GAIN_EPS for r in eligible),
            'spearman_with_ap_gain': correlation(x, y),
            'spearman_with_baseline_ap': correlation(x, [r['baseline_ap'] for r in eligible]),
            'auroc_beneficial_vs_rest': float(roc_auc_score(useful, x)) if len(useful) and useful.any() and (~useful).any() else None,
            'auroc_beneficial_vs_harmful': float(roc_auc_score(truth, [r['scores'][feature] for r in decisive]))
                if len(truth) and truth.any() and (~truth).any() else None}


def acceptance_summary(rows, accepted):
    """Scalar AP effects are exact for whole-image hard acceptance decisions."""
    selected = set(accepted)
    abnormal = [r for r in rows if r['ap_gain'] is not None]
    normal = [r for r in rows if r['anomaly'] == 0]
    gains = [r['ap_gain'] if r['index'] in selected else 0. for r in abnormal]
    categories = sorted({r['class'] for r in abnormal})
    macro = [mean([r['ap_gain'] if r['index'] in selected else 0. for r in abnormal if r['class'] == c]) for c in categories]
    valid_count = sum(r['valid'] for r in rows)
    probability = len(selected) / valid_count if valid_count else 0.
    random_macro = [mean([probability * r['ap_gain'] if r['valid'] else 0. for r in abnormal if r['class'] == c]) for c in categories]
    return {'accepted_images': len(selected), 'total_images': len(rows),
            'accepted_normal': sum(r['index'] in selected for r in normal),
            'accepted_anomalous': sum(r['index'] in selected for r in rows if r['anomaly'] == 1),
            'mean_anomalous_image_ap_gain': mean(gains),
            'class_balanced_anomalous_image_ap_gain': mean(macro),
            'same_count_random_expected_class_balanced_gain': mean(random_macro),
            'accepted_beneficial': sum(r['index'] in selected and r['ap_gain'] > GAIN_EPS for r in abnormal),
            'accepted_harmful': sum(r['index'] in selected and r['ap_gain'] < -GAIN_EPS for r in abnormal),
            'rejected_beneficial': sum(r['index'] not in selected and r['ap_gain'] > GAIN_EPS for r in abnormal),
            'rejected_harmful': sum(r['index'] not in selected and r['ap_gain'] < -GAIN_EPS for r in abnormal),
            'normal_mean_score_delta': mean([r['normal_mean_delta'] if r['index'] in selected else 0. for r in normal])}


def ranking_summary(rows):
    eligible = [r for r in rows if r['valid']]
    result = {'images': len(rows), 'valid_evidence_images': len(eligible),
              'images_with_defined_ap': sum(r['ap_gain'] is not None for r in rows),
              'associations': {f: associations(rows, f) for f in FEATURES},
              'by_class': {}, 'diagnostic_coverage_curves': {}, 'fixed_policies': {}}
    for name in sorted({r['class'] for r in rows}):
        group = [r for r in rows if r['class'] == name]
        result['by_class'][name] = {f: associations(group, f) for f in FEATURES}
    for feature in FEATURES:
        # No labels/outcomes in ordering. Ties use original metadata order.
        ordered = sorted(eligible, key=lambda r: (-r['scores'][feature], r['index']))
        result['diagnostic_coverage_curves'][feature] = []
        for fraction in COVERAGES:
            count = math.ceil(len(ordered) * fraction)
            selected = [r['index'] for r in ordered[:count]]
            result['diagnostic_coverage_curves'][feature].append(
                {'coverage_of_valid_images': fraction, **acceptance_summary(rows, selected)})
    for name in ('baseline', 'all_fusion', 'consistency', 'matched_score'):
        result['fixed_policies'][name] = acceptance_summary(rows, [r['index'] for r in rows if r['decisions'][name]])
    result['examples'] = {}
    for name, group in (
            ('accepted_consistency_harmful', [r for r in rows if r['decisions']['consistency'] and r['ap_gain'] is not None and r['ap_gain'] < -GAIN_EPS]),
            ('rejected_consistency_beneficial', [r for r in rows if not r['decisions']['consistency'] and r['ap_gain'] is not None and r['ap_gain'] > GAIN_EPS])):
        result['examples'][name] = sorted(group, key=lambda r: -abs(r['ap_gain']))[:8]
    return result


def diagnose_target(path, calibration):
    target = json.loads(path.read_text(encoding='utf8'))
    validate_inputs(calibration, target)
    root, classes = path.parent, list(target['metrics'])
    if set(classes) != {r['class'] for r in target['per_image']}:
        raise ValueError('Target class/cache index mismatch')
    positions = {r['image']: i for i, r in enumerate(target['per_image'])}
    if len(positions) != len(target['per_image']):
        raise ValueError('Duplicate target image identity')
    records = []
    side = target['fixed_rules']['config']['image_size']
    for i, name in enumerate(classes):
        rows = [r for r in target['per_image'] if r['class'] == name]
        if sorted(r['cache_position'] for r in rows) != list(range(len(rows))):
            raise ValueError('Invalid cache positions in ' + name)
        arrays = {k: np.load(safe_child(root, 'cache/class_%02d_%s.npy' % (i, k)), mmap_mode='r', allow_pickle=False)
                  for k in ('baseline', 'fused', 'mask')}
        if any(v.shape != (len(rows), side, side) for v in arrays.values()) or arrays['mask'].dtype != np.bool_:
            raise ValueError('Invalid target cache shape/dtype')
        for j, row in enumerate(rows):
            index = row['cache_position']
            mask, base, fused = (arrays[k][index] for k in ('mask', 'baseline', 'fused'))
            normal = row['anomaly'] == 0
            if normal and mask.any():
                raise ValueError('Normal image contains positive GT pixels')
            base_ap = image_ap(mask, base)
            # Validate fused even when no GT positives exist, then skip AP sorting.
            if not np.isfinite(fused).all() or np.min(fused) < 0 or np.max(fused) > 1:
                raise ValueError('Invalid fused probability map')
            fused_ap = (base_ap if np.array_equal(base, fused) else image_ap(mask, fused)) if base_ap is not None else None
            valid = row['evidence']['valid']
            # local_core_mean was not persisted in target reports; reconstruct it
            # from cached post-filter maps would change its definition. Therefore
            # use global image score as the third existing confidence comparator.
            scores = {'soft_iou': row['evidence']['soft_iou'],
                      'baseline_core_mean': row['evidence']['baseline_core_mean'],
                      'global_image_score': row['image_score']}
            if valid and any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in scores.values()):
                raise ValueError('Non-finite ranking score')
            records.append({'index': positions[row['image']], 'image': row['image'], 'class': name,
                'anomaly': row['anomaly'], 'valid': valid, 'scores': scores,
                'baseline_ap': base_ap, 'fused_ap': fused_ap,
                'ap_gain': fused_ap - base_ap if base_ap is not None else None,
                'normal_mean_delta': float(fused.mean() - base.mean()) if normal else None,
                'decisions': row['decisions']})
            if (j + 1) % 32 == 0:
                print('%s class=%s per-image AP=%d/%d' % (target['testing_data'], name, j + 1, len(rows)), flush=True)
        print('%s completed class=%s images=%d' % (target['testing_data'], name, len(rows)), flush=True)
    records.sort(key=lambda r: r['index'])
    return target, records


def main(args):
    source_path = Path(args.calibration_report).resolve()
    source = json.loads(source_path.read_text(encoding='utf8'))
    paths = [Path(p).resolve() for p in args.target_reports]
    if len(set(paths)) != len(paths):
        raise ValueError('Target reports must be distinct')
    output = Path(args.save_path).resolve()
    if any(output in p.parents or p.parent in output.parents for p in [source_path] + paths):
        raise ValueError('Use output outside input experiment directories')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Use a new/empty diagnosis directory')
    # Validate all metadata before computing per-image AP on either dataset.
    names = []
    for path in paths:
        target = json.loads(path.read_text(encoding='utf8'))
        validate_inputs(source, target)
        name = target['testing_data']
        if not isinstance(name, str) or not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name):
            raise ValueError('Unsafe target dataset name')
        if not target['per_image']:
            raise ValueError('Target report has no images')
        classes = list(target['metrics'])
        if set(classes) != {r['class'] for r in target['per_image']}:
            raise ValueError('Target class/cache index mismatch')
        for i in range(len(classes)):
            for key in ('baseline', 'fused', 'mask'):
                safe_child(path.parent, 'cache/class_%02d_%s.npy' % (i, key))
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError('One report per target dataset is required')
    output.mkdir(parents=True, exist_ok=True)
    summary = {'status': 'running', 'no_model_inference': True, 'deployment_rules_changed': False,
        'source_report_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'gain_epsilon': GAIN_EPS, 'metrics_units': '0..1 except original_dataset_metrics (0..100)',
        'coverage_points_predeclared': list(COVERAGES),
        'score_definitions': {'soft_iou': 'spatial consistency of two aligned crop predictions',
            'baseline_core_mean': 'initial global-map confidence within the candidate core',
            'global_image_score': 'original global AdaCLIP/HSF image anomaly score'},
        'ap_eligibility': 'Per-image AP requires both positive and negative GT pixels; normal images '
                          'have undefined AP but remain in coverage selection and normal-score diagnostics.',
        'note': 'Per-image AP utility differs from category-pooled dataset AP. Coverage curves are '
                'GT-evaluated diagnostics of fixed score rankings across ALL valid images, including '
                'normal images; they do not choose target deployment thresholds. AUROC here predicts '
                'whether a crop update improves AP, NOT whether an image is anomalous. Normal score '
                'changes are not thresholded false-positive counts.',
        'datasets': {}}
    for path in paths:
        target, rows = diagnose_target(path, source)
        name = target['testing_data']
        diagnostic = ranking_summary(rows)
        diagnostic['target_report_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        diagnostic['original_dataset_metrics'] = target['average']
        diagnostic['fixed_rules'] = {k: target['fixed_rules'][k] for k in ('consistency', 'matched_score')}
        summary['datasets'][name] = diagnostic
        (output / (name + '_per_image_utility.json')).write_text(json.dumps(rows, indent=2, allow_nan=False), encoding='utf8')
        (output / 'reinspection_ranking_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf8')
    summary['status'] = 'complete'
    (output / 'reinspection_ranking_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf8')
    lines = ['Reinspection ranking diagnosis DONE', 'No inference or target-based deployment threshold tuning.',
             'AUROC below measures beneficial-update ranking, not anomaly detection.']
    for name, d in summary['datasets'].items():
        lines.append('%s images=%d valid=%d defined_AP=%d' % (name, d['images'], d['valid_evidence_images'], d['images_with_defined_ap']))
        for feature, a in d['associations'].items():
            lines.append('  %s AP-gain Spearman=%s gain-AUROC=%s useful-vs-harmful-AUROC=%s baseline-AP-Spearman=%s' % (
                feature, a['spearman_with_ap_gain'], a['auroc_beneficial_vs_rest'],
                a['auroc_beneficial_vs_harmful'], a['spearman_with_baseline_ap']))
        for feature, points in d['diagnostic_coverage_curves'].items():
            for p in points:
                lines.append('  DIAGNOSTIC %s coverage=%.2f accepted=%d normal=%d macro_AP_gain=%s random_expected=%s' % (
                    feature, p['coverage_of_valid_images'], p['accepted_images'], p['accepted_normal'],
                    p['class_balanced_anomalous_image_ap_gain'], p['same_count_random_expected_class_balanced_gain']))
    text = '\n'.join(lines) + '\n'
    (output / 'reinspection_ranking_summary.txt').write_text(text, encoding='utf8')
    print(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration_report', required=True)
    parser.add_argument('--target_reports', nargs='+', required=True)
    parser.add_argument('--save_path', required=True)
    main(parser.parse_args())
