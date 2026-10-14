"""Fit crop acceptance thresholds on source calibration identities only.

Replays per-image fixed-fusion metrics from completed reports; no CLIP imports,
GPU inference, target data, or validation-based threshold tuning. Macro metrics
here average per-image metrics within category, then average categories.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


PREDICTORS = {'consistency': 'soft_iou', 'initial_score': 'baseline_core_mean'}
CONFIG_KEYS = ('model', 'image_size', 'use_hsf', 'k_clusters', 'prompting_type',
               'prompting_branch', 'prompting_depth', 'prompting_length', 'baseline_sha256',
               'seed', 'crop_fraction', 'shift_fraction', 'border_fraction', 'candidate_selector',
               'prompts', 'contrast', 'core_side_fraction', 'outer_side_fraction',
               'candidate_fusion_weight', 'metrics', 'invalid_evidence_policy')
MODES = ('baseline', 'single_crop_fixed_fusion', 'two_crop_fixed_fusion')
METRICS = ('pixel_ap', 'pixel_auroc', 'background_mean', 'background_q99')
SAFETY_EPS = 1e-12
GAIN_EPS = 1e-4  # 0.01 percentage point, same diagnostic definition as previous report.


def mean(values):
    return sum(values) / len(values) if values else None


def accepts(row, rule):
    """Decision depends ONLY on evidence. Labels/outcomes never enter this path."""
    if not row['evidence']['valid'] or rule['kind'] == 'reject_all':
        return False
    if rule['kind'] == 'accept_all':
        return True
    return row['evidence'][rule['feature']] >= rule['threshold']


def match_initial_acceptance(rows, consistency_rule):
    """Match MVTec CALIBRATION coverage, without validation or outcome input."""
    rows = [r for r in rows if r['source'] == 'mvtec']
    target = sum(accepts(r, consistency_rule) for r in rows)
    scores = sorted({r['evidence']['baseline_core_mean'] for r in rows if r['evidence']['valid']})
    choices = [{'kind': 'reject_all', 'feature': 'baseline_core_mean'},
               {'kind': 'accept_all', 'feature': 'baseline_core_mean'}]
    choices.extend({'kind': 'threshold', 'feature': 'baseline_core_mean', 'threshold': float(s)} for s in scores)
    counts = [sum(accepts(r, rule) for r in rows) for rule in choices]
    index = min(range(len(choices)), key=lambda i: (abs(counts[i] - target), counts[i]))
    return {'selected_rule': choices[index], 'calibration_target_count': target,
            'calibration_matched_count': counts[index], 'calibration_images': len(rows)}


def policy_summary(rows, probabilities, candidate='two_crop_fixed_fusion'):
    """Probabilities allow an exact random-acceptance expectation, without fitting."""
    if len(rows) != len(probabilities):
        raise ValueError('Policy size mismatch')
    output = {'images': len(rows), 'expected_accepted_images': sum(probabilities),
              'acceptance_fraction': mean(probabilities), 'by_label': {}, 'per_class': {},
              'class_balanced': {}, 'image_average': {}}
    records = []
    for row, probability in zip(rows, probabilities):
        if not 0 <= probability <= 1:
            raise ValueError('Invalid acceptance probability')
        values = {}
        for metric in METRICS:
            base, proposed = row['metrics']['baseline'][metric], row['metrics'][candidate][metric]
            values[metric] = base + probability * (proposed - base) if base is not None and proposed is not None else None
        records.append((row, probability, values))
    for label in (0, 1):
        group = [(r, p, v) for r, p, v in records if r['anomaly'] == label]
        output['by_label'][str(label)] = {'images': len(group), 'expected_accepted_images': sum(p for _, p, _ in group),
                                        'acceptance_fraction': mean([p for _, p, _ in group])}
        for metric in METRICS:
            vals = [v[metric] for _, _, v in group if v[metric] is not None]
            output['by_label'][str(label)][metric] = mean(vals)
    for category in sorted({r['source'] + ':' + r['class'] for r in rows}):
        group = [(r, p, v) for r, p, v in records if r['source'] + ':' + r['class'] == category]
        output['per_class'][category] = {'images': len(group), 'expected_accepted_images': sum(p for _, p, _ in group)}
        for label in (0, 1):
            selected = [v for r, _, v in group if r['anomaly'] == label]
            output['per_class'][category][str(label)] = {
                metric: mean([v[metric] for v in selected if v[metric] is not None]) for metric in METRICS}
    for label in (0, 1):
        output['class_balanced'][str(label)] = {}
        for metric in METRICS:
            vals = [g[str(label)][metric] for g in output['per_class'].values() if g[str(label)][metric] is not None]
            output['class_balanced'][str(label)][metric] = mean(vals)
    for metric in METRICS:
        output['image_average'][metric] = mean([v[metric] for _, _, v in records if v[metric] is not None])
    abnormal = [(r, p) for r, p, _ in records if r['outcomes']['two_crop_ap_delta'] is not None]
    output['utility_counts_two_crop_candidate'] = {
        'beneficial_updates_accepted': sum(p for r, p in abnormal if r['outcomes']['two_crop_ap_delta'] > GAIN_EPS),
        'harmful_updates_accepted': sum(p for r, p in abnormal if r['outcomes']['two_crop_ap_delta'] < -GAIN_EPS),
        'beneficial_updates_rejected': sum(1 - p for r, p in abnormal if r['outcomes']['two_crop_ap_delta'] > GAIN_EPS),
        'harmful_updates_rejected': sum(1 - p for r, p in abnormal if r['outcomes']['two_crop_ap_delta'] < -GAIN_EPS)}
    if candidate == 'single_crop_fixed_fusion':
        output.pop('utility_counts_two_crop_candidate')
    return output


def fit_threshold(rows, feature):
    """Fit on MVTec source calibration subset. Explicit no-useful-rule fallback.

    Maximize class-balanced anomalous-image AP gain, subject to BOTH normal
    response means (mean and q99, category balanced) not increasing. This is
    source-label calibration, not a label-free guarantee on future inputs.
    """
    rows = [r for r in rows if r['source'] == 'mvtec']
    if not rows or not any(r['anomaly'] == 0 for r in rows) or not any(r['anomaly'] == 1 for r in rows):
        raise ValueError('Calibration requires MVTec normal and anomalous source images')
    baseline = policy_summary(rows, [0.] * len(rows))['class_balanced']
    thresholds = sorted({r['evidence'][feature] for r in rows if r['evidence']['valid']})
    choices = [{'kind': 'reject_all', 'feature': feature}, {'kind': 'accept_all', 'feature': feature}]
    choices.extend({'kind': 'threshold', 'feature': feature, 'threshold': float(t)} for t in thresholds)
    evaluations = []
    for rule in choices:
        flags = [float(accepts(r, rule)) for r in rows]
        summary = policy_summary(rows, flags)
        macro = summary['class_balanced']
        gain = macro['1']['pixel_ap'] - baseline['1']['pixel_ap']
        normal_delta = {k: macro['0'][k] - baseline['0'][k] for k in ('background_mean', 'background_q99')}
        safe = all(d <= SAFETY_EPS for d in normal_delta.values())
        evaluations.append({'rule': rule, 'anomalous_ap_gain': gain, 'normal_response_delta': normal_delta,
                            'satisfies_normal_constraints': safe, 'summary': summary})
    feasible = [e for e in evaluations if e['satisfies_normal_constraints']]
    # Reject-all is always feasible. On equal utility prefer fewer harmful
    # updates, then greater coverage; all tie-breaking uses calibration data.
    best = max(feasible, key=lambda e: (e['anomalous_ap_gain'],
               -e['summary']['utility_counts_two_crop_candidate']['harmful_updates_accepted'],
               e['summary']['expected_accepted_images']))
    if best['anomalous_ap_gain'] <= SAFETY_EPS:
        best = evaluations[0]
    return {'feature': feature, 'status': 'useful_calibration_rule' if best['anomalous_ap_gain'] > SAFETY_EPS else 'no_useful_rule',
            'selected_rule': best['rule'], 'selected_result': best,
            'searched_rules': len(evaluations), 'feasible_rules': len(feasible)}


def validate_report(report, expected_rows, label):
    if report.get('status') != 'complete' or report.get('unchanged_detector_tensors', 0) <= 0:
        raise ValueError(label + ': need a complete report with a passed frozen-detector check')
    expected = {str(Path(r['image']).resolve()): (r['source'], r['class'], r['anomaly']) for r in expected_rows}
    if len(expected) != len(expected_rows):
        raise ValueError(label + ': duplicate reference identity')
    rows = report['per_image']
    actual = {}
    for r in rows:
        path = str(Path(r['image']).resolve())
        identity = (r['source'], r['class'], r['anomaly'])
        if path in actual or expected.get(path) != identity:
            raise ValueError(label + ': duplicate or wrong split identity')
        actual[path] = identity
        for mode in MODES:
            for key in METRICS:
                value = r['metrics'][mode][key]
                if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError(label + ': invalid per-image metric')
        if r['evidence']['valid']:
            for key in PREDICTORS.values():
                value = r['evidence'][key]
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(label + ': invalid predictor score')
        expected_delta = r['metrics']['two_crop_fixed_fusion']['pixel_ap']
        baseline = r['metrics']['baseline']['pixel_ap']
        expected_delta = expected_delta - baseline if expected_delta is not None and baseline is not None else None
        recorded = r['outcomes']['two_crop_ap_delta']
        if (expected_delta is None) != (recorded is None) or (expected_delta is not None and not math.isclose(expected_delta, recorded, abs_tol=1e-10)):
            raise ValueError(label + ': AP delta differs from underlying metrics')
    if set(actual) != set(expected):
        raise ValueError(label + ': report must cover the EXACT original split')


def validate_pair(calibration, validation, reference):
    if calibration['metadata'].get('split') != 'train':
        raise ValueError('Calibration must be explicitly generated with --split train')
    if validation['metadata'].get('split', 'validation') != 'validation':
        raise ValueError('Comparison input must be the validation report')
    training_ids = {str(Path(r['image']).resolve()) for r in reference['train']['per_image']}
    validation_ids = {str(Path(r['image']).resolve()) for r in reference['validation']['per_image']}
    if training_ids & validation_ids:
        raise ValueError('Original reference splits overlap')
    validate_report(calibration, reference['train']['per_image'], 'calibration')
    validate_report(validation, reference['validation']['per_image'], 'validation')
    for key in CONFIG_KEYS:
        if key not in calibration['metadata'] or calibration['metadata'].get(key) != validation['metadata'].get(key):
            raise ValueError('Calibration/validation configuration mismatch: ' + key)
    for report in (calibration, validation):
        if Path(report['metadata']['baseline_checkpoint']).resolve() != Path(reference['metadata']['baseline_checkpoint']).resolve():
            raise ValueError('Baseline checkpoint differs from original reference')


def run_calibration(calibration_path, validation_path, reference_path, save_path):
    calibration_path, validation_path, reference_path = map(Path, (calibration_path, validation_path, reference_path))
    destination = Path(save_path).resolve()
    if any(destination in p.resolve().parents for p in (calibration_path, validation_path, reference_path)):
        raise ValueError('Comparison output must be separate from input directories')
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError('Use new/empty comparison output directory')
    calibration = json.loads(calibration_path.read_text(encoding='utf8'))
    validation = json.loads(validation_path.read_text(encoding='utf8'))
    reference = json.loads(reference_path.read_text(encoding='utf8'))
    validate_pair(calibration, validation, reference)
    # ONLY calibration rows are passed into threshold fitting.
    fitted = {name: fit_threshold(calibration['per_image'], key) for name, key in PREDICTORS.items()}
    destination.mkdir(parents=True, exist_ok=True)
    rules = {'status': 'complete', 'calibration_source': 'mvtec',
             'calibration_images_mvtec': sum(r['source'] == 'mvtec' for r in calibration['per_image']),
             'objective': 'maximize class-balanced anomalous per-image AP gain',
             'constraints': 'class-balanced normal background mean AND q99 do not increase',
             'numerical_safety_slack': SAFETY_EPS,
             'fallback': 'reject all when no positive-gain feasible rule exists',
             'source_labels_used_for_calibration': True, 'no_validation_threshold_selection': True,
             'calibration_report_sha256': hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
             'config': {k: calibration['metadata'][k] for k in CONFIG_KEYS}, 'rules': fitted}
    (destination / 'calibration_rules.json').write_text(json.dumps(rules, indent=2, allow_nan=False), encoding='utf8')
    result = {'status': 'complete', 'validation_report': str(validation_path.resolve()),
              'rules_file': str(destination / 'calibration_rules.json'),
              'note': 'Exploratory source validation. Per-image category averages, not official pooled dataset AP. '
                      'All selector policies use the SAME two computed crop predictions; no compute-saving claim.',
              'by_source': {}, 'per_image': []}
    for source in reference['metadata']['sources']:
        rows = [r for r in validation['per_image'] if r['source'] == source]
        if not rows:
            continue
        policies = {'baseline': policy_summary(rows, [0.] * len(rows)),
                    'single_crop_fixed_fusion': policy_summary(rows, [float(r['evidence']['valid']) for r in rows], 'single_crop_fixed_fusion'),
                    'all_two_crop_fixed_fusion': policy_summary(rows, [float(r['evidence']['valid']) for r in rows])}
        eligible = sum(r['evidence']['valid'] for r in rows)
        for name, fitted_rule in fitted.items():
            flags = [float(accepts(r, fitted_rule['selected_rule'])) for r in rows]
            policies[name] = policy_summary(rows, flags)
            rate = sum(flags) / eligible if eligible else 0.
            policies[name + '_random_matched_expectation'] = policy_summary(rows, [rate if r['evidence']['valid'] else 0. for r in rows])
        result['by_source'][source] = policies
    for row in validation['per_image']:
        result['per_image'].append({k: row[k] for k in ('source', 'class', 'image', 'anomaly')})
        result['per_image'][-1]['decisions'] = {name: accepts(row, rule['selected_rule']) for name, rule in fitted.items()}
    (destination / 'validation_policy_comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    lines = ['Reinspection calibration DONE', 'Rules fitted on MVTec train subset only; fixed before validation comparison.']
    for name, fit in fitted.items():
        lines.append(name + ': ' + fit['status'] + ' rule=' + json.dumps(fit['selected_rule']))
    for source, policies in result['by_source'].items():
        for name, summary in policies.items():
            lines.append('%s %s accepted=%s/%d macro_abnormal_AP=%s macro_normal_mean=%s macro_normal_q99=%s' % (
                source, name, summary['expected_accepted_images'], summary['images'],
                summary['class_balanced']['1']['pixel_ap'], summary['class_balanced']['0']['background_mean'],
                summary['class_balanced']['0']['background_q99']))
    text = '\n'.join(lines) + '\n'
    (destination / 'validation_policy_comparison.txt').write_text(text, encoding='utf8')
    print(text)
    return rules, result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration_report', required=True)
    parser.add_argument('--validation_report', required=True)
    parser.add_argument('--reference_report', required=True)
    parser.add_argument('--save_path', required=True)
    args = parser.parse_args()
    run_calibration(args.calibration_report, args.validation_report, args.reference_report, args.save_path)
