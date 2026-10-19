"""Offline, GT-assisted single-crop utility diagnosis; no model/GPU or rule fitting.

Whole-image choices deliberately use target GT, so they are NOT deployable
policies, formal benchmark gains, or upper bounds on category-pooled AP.
Only P-AP and one source-fixed pixel operating point are evaluated here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from audit_reinspection_risk import (accumulate, calibrate_pixel_threshold,
    empty_counts, finish, pixel_counts, safe_child)
from calibrate_reinspection import CONFIG_KEYS
from diagnose_reinspection_ranking import image_ap, mean
from diagnose_selection_effects import GAIN_EPS, changes, pooled_ap, validate_report


ACTUAL = ('baseline', 'all_single')
DIAGNOSTIC = ('gt_ap_all_normals', 'gt_ap_base_normals', 'gt_ap_fp_normals')
POLICIES = ACTUAL + DIAGNOSTIC


def validate_inputs(source, target, source_path):
    if (source.get('status') != 'complete' or source['metadata'].get('split') != 'train'
            or source.get('unchanged_detector_tensors', 0) <= 0
            or source.get('initial_equivalence', {}).get('passed') is not True):
        raise ValueError('Require completed train-source calibration with frozen/equivalence PASS')
    config = target['protocol']['source_config']
    for key in CONFIG_KEYS:
        if key not in source['metadata'] or key not in config or source['metadata'][key] != config[key]:
            raise ValueError('Calibration/target configuration mismatch: ' + key)
    if target['protocol'].get('source_report_sha256') != hashlib.sha256(source_path.read_bytes()).hexdigest():
        raise ValueError('Calibration report hash differs from target source provenance')
    guard = target.get('reference_equivalence', {})
    if guard.get('enabled') is not True or guard.get('passed') is not True:
        raise ValueError('Single-crop reference equivalence must be enabled and passed')
    source_ids = [str(Path(r['image']).resolve()) for r in source['per_image']]
    target_ids = {str(Path(r['image']).resolve()) for r in target['per_image']}
    if not source_ids or len(set(source_ids)) != len(source_ids) or set(source_ids) & target_ids:
        raise ValueError('Duplicate or overlapping source/target image identities')


def diagnostic_choices(anomaly, base_ap, single_ap, base_counts, single_counts):
    # Strict AP improvement; the reporting epsilon NEVER controls selection.
    improve = base_ap is not None and single_ap is not None and single_ap > base_ap
    if anomaly == 1:
        return {p: improve for p in DIAGNOSTIC}
    return dict(gt_ap_all_normals=True, gt_ap_base_normals=False,
                gt_ap_fp_normals=single_counts['fp'] < base_counts['fp'])


def utility_summary(records):
    anomalous = [r for r in records if r['anomaly'] == 1]
    defined = [r for r in anomalous if r['ap_delta'] is not None]
    normal = [r for r in records if r['anomaly'] == 0]
    deltas = [r['ap_delta'] for r in defined]
    normal_deltas = [r['counts']['all_single']['fp'] - r['counts']['baseline']['fp'] for r in normal]
    return dict(images=len(records), normal_images=len(normal), anomalous_images=len(anomalous),
        anomalous_images_without_defined_ap=len(anomalous) - len(defined),
        anomalous_image_ap_changes=changes(deltas),
        strictly_positive_ap_updates=sum(d > 0 for d in deltas),
        per_image_ap_mean={p: mean([r['single_ap'] if r['choose_single'][p] else r['base_ap']
                                    for r in defined]) for p in POLICIES},
        mean_positive_part_ap_gain=mean([max(0., d) for d in deltas]),
        normal_fp_changes=dict(increased=sum(d > 0 for d in normal_deltas),
            decreased=sum(d < 0 for d in normal_deltas), unchanged=sum(d == 0 for d in normal_deltas),
            new_any_positive_images=sum(r['counts']['baseline']['fp'] == 0 and r['counts']['all_single']['fp'] > 0 for r in normal),
            removed_any_positive_images=sum(r['counts']['baseline']['fp'] > 0 and r['counts']['all_single']['fp'] == 0 for r in normal)))


def diagnose_class(path, target, index, category, threshold, scratch):
    rows = [r for r in target['per_image'] if r['class'] == category]
    cache = {m: np.load(safe_child(path.parent, 'cache/class_%02d_%s.npy' % (index, m)),
                       mmap_mode='r', allow_pickle=False) for m in ('baseline', 'adaptive', 'mask')}
    if any(cache[m].dtype != np.float32 for m in ('baseline', 'adaptive')):
        raise ValueError('Expected float32 target prediction caches')
    mixtures = {p: np.lib.format.open_memmap(scratch / (p + '.npy'), mode='w+',
        dtype=np.float32, shape=cache['mask'].shape) for p in DIAGNOSTIC}
    totals = {p: empty_counts() for p in POLICIES}
    records = []
    for j, row in enumerate(rows):
        position, normal = row['cache_position'], row['anomaly'] == 0
        mask, base, single = (cache[m][position] for m in ('mask', 'baseline', 'adaptive'))
        if normal and mask.any():
            raise ValueError('Normal image has positive GT pixels')
        geometry = row['selections']['adaptive']
        if type(geometry.get('geometry_valid')) is not bool or geometry.get('crop_forwards') != 1:
            raise ValueError('Invalid primary-A single-crop record')
        if not geometry['geometry_valid'] and not np.array_equal(base, single):
            raise ValueError('Invalid crop geometry must preserve baseline exactly')
        base_ap, single_ap = image_ap(mask, base), image_ap(mask, single)
        counts = dict(baseline=pixel_counts(mask, base, threshold),
                      all_single=pixel_counts(mask, single, threshold))
        choices = dict(baseline=False, all_single=True,
            **diagnostic_choices(row['anomaly'], base_ap, single_ap, counts['baseline'], counts['all_single']))
        for policy in POLICIES:
            selected = 'all_single' if choices[policy] else 'baseline'
            accumulate(totals[policy], counts[selected], normal)
            if policy in DIAGNOSTIC:
                mixtures[policy][position] = single if choices[policy] else base
        records.append(dict(image=row['image'], **{'class': category}, anomaly=row['anomaly'],
            cache_position=position, foreground_pixels=int(mask.sum()), geometry_valid=geometry['geometry_valid'],
            base_ap=base_ap, single_ap=single_ap,
            ap_delta=single_ap - base_ap if base_ap is not None and single_ap is not None else None,
            choose_single=choices, counts=counts))
        if (j + 1) % 32 == 0:
            print('%s class=%s per-image=%d/%d' % (target['testing_data'], category, j + 1, len(rows)), flush=True)
    scores = {}
    for policy in POLICIES:
        print('%s class=%s pooled P-AP=%s' % (target['testing_data'], category, policy), flush=True)
        if policy in ACTUAL:
            mode = 'baseline' if policy == 'baseline' else 'adaptive'
            values = cache[mode]
        else:
            mixtures[policy].flush()
            values = mixtures[policy]
        scores[policy] = pooled_ap(cache['mask'], values)
        if policy in ACTUAL and not math.isclose(scores[policy], target['metrics'][category][mode]['ap_px'],
                                                abs_tol=1e-6, rel_tol=0):
            raise ValueError('Cached/reported pooled AP mismatch: %s/%s' % (category, mode))
    result = dict(**utility_summary(records), pooled_ap_percent=scores,
        pooled_ap_delta_vs_baseline_pp={p: scores[p] - scores['baseline'] for p in POLICIES},
        pooled_ap_delta_vs_all_single_pp={p: scores[p] - scores['all_single'] for p in POLICIES},
        pixel_operating_point={p: finish(totals[p]) for p in POLICIES})
    # Release memmaps before TemporaryDirectory cleanup (also works on Windows).
    del values, mixtures
    return result, records


def main(args):
    source_path, target_path = Path(args.calibration_report).resolve(), Path(args.target_report).resolve()
    output = Path(args.save_path).resolve()
    for path in (source_path, target_path):
        if output == path.parent or output in path.parents or path.parent in output.parents:
            raise ValueError('Use output separate from input experiment directories')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Use a new/empty utility diagnosis directory')
    if not 0 < args.source_pixel_fpr < 1:
        raise ValueError('source_pixel_fpr must be in (0,1)')
    source = json.loads(source_path.read_text(encoding='utf8'))
    target = validate_report(target_path, single_crop=True)
    validate_inputs(source, target, source_path)
    output.mkdir(parents=True, exist_ok=True)
    print('Calibrating a SOURCE-normal-only pixel threshold; no model inference.', flush=True)
    operating = calibrate_pixel_threshold(source, source_path.parent, output, args.source_pixel_fpr)
    result = dict(status='running', testing_data=target['testing_data'], no_model_inference=True,
        deployment_rules_changed=False, target_gt_used_for_diagnostic_choices=True,
        not_a_pooled_ap_upper_bound=True, source_report_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        target_report_sha256=hashlib.sha256(target_path.read_bytes()).hexdigest(), operating_point=operating,
        protocol=dict(no_second_smoothing=True, all_images_in_pooled_ap=True, no_parameter_search=True,
            metrics='category-pooled P-AP (0..100), macro category mean; fixed-source-threshold pixel counts/rates',
            image_ap_units='0..1; undefined for all-background/all-foreground masks',
            reporting_gain_epsilon=GAIN_EPS, choice_gain_epsilon=0.,
            anomalous_choice='Use single iff its per-image AP is strictly larger; baseline on ties/undefined AP',
            policies=dict(baseline='Original global map', all_single='Existing adaptive single-A fixed fusion',
                gt_ap_all_normals='GT AP choice on anomalous images; single on every normal image',
                gt_ap_base_normals='GT AP choice on anomalous images; baseline on every normal image',
                gt_ap_fp_normals='GT AP choice on anomalous images; on normals use single only if fixed-threshold FP strictly decreases'),
            caveats='All gt_* choices leak target GT and are diagnostic only, not deployable gates or formal test gains. '
                'Per-image AP choices need not maximize pooled AP because of cross-image ranks and pixel weights. '
                'Normal any-positive fraction is NOT HSF image-classifier FPR. Threshold diagnostics are not max-F1. '
                'No claim of latency savings; selection here observes already-computed crop predictions.'), per_class={})
    summary_path = output / 'single_reinspection_utility.json'
    summary_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    records = []
    for i, category in enumerate(target['metrics']):  # Cache order is NOT alphabetical.
        with tempfile.TemporaryDirectory(prefix='utility_scratch_', dir=output) as temporary:
            diagnostic, rows = diagnose_class(target_path, target, i, category,
                operating['pixel_threshold'], Path(temporary))
        result['per_class'][category] = diagnostic
        records.extend(rows)
        print('Completed category=%s images=%d' % (category, len(rows)), flush=True)
        summary_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    result['summary'] = utility_summary(records)
    result['class_average_pooled_ap_percent'] = {p: mean([c['pooled_ap_percent'][p]
        for c in result['per_class'].values()]) for p in POLICIES}
    result['class_balanced_per_image_ap'] = {p: mean([c['per_image_ap_mean'][p]
        for c in result['per_class'].values() if c['per_image_ap_mean'][p] is not None]) for p in POLICIES}
    for p, mode in (('baseline', 'baseline'), ('all_single', 'adaptive')):
        if not math.isclose(result['class_average_pooled_ap_percent'][p], target['average'][mode]['ap_px'],
                            abs_tol=1e-6, rel_tol=0):
            raise ValueError('Cached/reported average AP mismatch')
    global_counts = {p: empty_counts() for p in POLICIES}
    for row in records:
        for p in POLICIES:
            accumulate(global_counts[p], row['counts']['all_single' if row['choose_single'][p] else 'baseline'], row['anomaly'] == 0)
    result['pixel_pooled_operating_point'] = {p: finish(global_counts[p]) for p in POLICIES}
    defined = [r for r in records if r['ap_delta'] is not None]
    result['examples'] = dict(harmful=sorted([r for r in defined if r['ap_delta'] < -GAIN_EPS], key=lambda r: r['ap_delta'])[:12],
        beneficial=sorted([r for r in defined if r['ap_delta'] > GAIN_EPS], key=lambda r: r['ap_delta'], reverse=True)[:12],
        normal_fp_increased=sorted([r for r in records if r['anomaly'] == 0 and r['counts']['all_single']['fp'] > r['counts']['baseline']['fp']],
            key=lambda r: r['counts']['all_single']['fp'] - r['counts']['baseline']['fp'], reverse=True)[:12])
    (output / 'single_reinspection_utility_per_image.json').write_text(json.dumps(records, indent=2, allow_nan=False), encoding='utf8')
    result['status'] = 'complete'
    summary_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    lines = ['Single reinspection utility diagnosis DONE',
        'GT-assisted diagnostic only; NOT deployable, formal benchmark gains, or a pooled AP upper bound.',
        'Source-only pixel threshold=%g; target maps are NOT smoothed again.' % operating['pixel_threshold'],
        'All pooled P-AP values include normal images. Per-image AP units=0..1; pooled P-AP units=percent.',
        'Overall image utility: ' + json.dumps(result['summary'])]
    for p in POLICIES:
        score = result['class_average_pooled_ap_percent'][p]
        counts = result['pixel_pooled_operating_point'][p]
        lines.append('%s P-AP=%.6f delta_vs_baseline=%+.6f pp delta_vs_all_single=%+.6f pp normal_pixel_FPR=%s recall=%s' % (
            p, score, score - result['class_average_pooled_ap_percent']['baseline'],
            score - result['class_average_pooled_ap_percent']['all_single'], counts['normal_pixel_fpr'], counts['pixel_recall']))
    for category, diagnostic in result['per_class'].items():
        lines.append('%s pooled_AP=%s image_AP_changes=%s normal_FP_changes=%s' % (category,
            json.dumps(diagnostic['pooled_ap_percent']), json.dumps(diagnostic['anomalous_image_ap_changes']),
            json.dumps(diagnostic['normal_fp_changes'])))
    text = '\n'.join(lines) + '\n'
    (output / 'single_reinspection_utility.txt').write_text(text, encoding='utf8')
    print(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration_report', required=True)
    parser.add_argument('--target_report', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--source_pixel_fpr', type=float, default=.001)
    main(parser.parse_args())
