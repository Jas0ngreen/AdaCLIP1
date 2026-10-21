"""Source-only offline pilot for predicting single-A reinspection utility.

Fit ONE fixed logistic classifier on MVTec source-train data; evaluate it on
the exact original source-validation identities. No CLIP/torch, target inputs,
threshold search, crop B prediction, model changes, or latency-saving claim.
Labels are utility proxies, not a guarantee about category-pooled AP.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import expit
from scipy.stats import rankdata
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from audit_reinspection_risk import (accumulate, calibrate_pixel_threshold, empty_counts,
                                    finish, pixel_counts, safe_child)
from calibrate_reinspection import CONFIG_KEYS
from diagnose_reinspection_ranking import mean
from diagnose_selection_effects import pooled_ap
from source_background_alignment import check_replay, metrics, prepare


FEATURES = ('baseline_support_mean', 'rank_correlation', 'low_response_shift',
            'absolute_update', 'concentration_change', 'q99_change')
SEEDS = (111, 222, 333, 444, 555)
AP_EPS = 1e-4
POLICIES = ('baseline', 'all_single', 'selector', 'confidence_matched') + tuple('random_%d' % s for s in SEEDS)


def prediction_features(base, single, support):
    """ONLY aligned, once-smoothed prediction maps and geometry; no GT/metadata."""
    if (base.ndim != 2 or single.shape != base.shape or support.shape != base.shape
            or support.dtype != np.bool_):
        raise ValueError('Invalid prediction/support shapes or dtype')
    for values in (base, single):
        if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
            raise ValueError('Invalid prediction probabilities')
    if not support.any():
        return [0.] * len(FEATURES)
    b, s = base[support].astype(np.float64), single[support].astype(np.float64)
    rb, rs = rankdata(b), rankdata(s)
    rb, rs = rb - rb.mean(), rs - rs.mean()
    denominator = float(np.linalg.norm(rb) * np.linalg.norm(rs))
    correlation = float(np.clip(np.dot(rb, rs) / denominator, -1, 1)) if denominator else 0.
    low = b <= np.median(b)  # All ties included; prediction-only, NOT known-normal pixels.
    def concentration(v):
        k = max(1, math.ceil(.1 * len(v)))
        return float(np.partition(v, len(v) - k)[-k:].sum() / v.sum()) if v.sum() else 0.
    return [float(b.mean()), correlation, float(np.median((s - b)[low])),
            float(np.abs(s - b).mean()), concentration(s) - concentration(b),
            float(np.quantile(s, .99) - np.quantile(b, .99))]


def utility_label(anomaly, ap_delta, fp_delta):
    if anomaly == 0:
        return 1 if fp_delta < 0 else 0 if fp_delta > 0 else None
    if anomaly != 1:
        raise ValueError('Invalid anomaly label')
    return (1 if ap_delta > AP_EPS else 0 if ap_delta < -AP_EPS else None) if ap_delta is not None else None


def score_features(model, features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
        raise ValueError('Invalid feature matrix')
    z = (x - np.asarray(model['scaler_mean'])) / np.asarray(model['scaler_scale'])
    return expit(z @ np.asarray(model['coefficient']) + model['intercept'])


def fit_selector(rows):
    eligible = [r for r in rows if r['geometry_valid'] and r['utility_label'] is not None]
    if not eligible or {r['utility_label'] for r in eligible} != {0, 1}:
        raise ValueError('Source train needs both beneficial and harmful eligible updates; no fitted selector produced')
    x = np.asarray([r['features'] for r in eligible], dtype=np.float64)
    y = np.asarray([r['utility_label'] for r in eligible], dtype=np.int64)
    groups = [(r['class'], r['anomaly']) for r in eligible]
    sizes = {g: groups.count(g) for g in set(groups)}
    weights = np.asarray([1. / sizes[g] for g in groups])
    weights *= len(weights) / weights.sum()
    scaler = StandardScaler().fit(x)  # TRAIN only, including preprocessing.
    classifier = LogisticRegression(C=1., solver='liblinear', tol=1e-6, max_iter=1000,
                                    random_state=111, class_weight=None)
    classifier.fit(scaler.transform(x), y, sample_weight=weights)
    if classifier.n_iter_.max() >= 1000:
        raise RuntimeError('Source-only classifier did not converge')
    model = dict(feature_names=list(FEATURES), scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(), coefficient=classifier.coef_[0].tolist(),
        intercept=float(classifier.intercept_[0]), acceptance_threshold=.5,
        positive_rule='geometry valid AND sigmoid(standardized_features @ coefficient + intercept) >= 0.5',
        classifier=dict(kind='L2 logistic regression', C=1., solver='liblinear', tol=1e-6,
            max_iter=1000, random_state=111, class_weight=None, sklearn_version=sklearn.__version__),
        fit_images=len(eligible), beneficial_fit_images=int(y.sum()), harmful_fit_images=int((y == 0).sum()),
        training_weight='inverse eligible train (class, anomaly) group size; normalized to sum=N',
        utility_proxy='anomalous AP delta > 1e-4 vs < -1e-4; normal source-fixed-threshold FP decrease vs increase; ties/undefined excluded',
        utility_probability_is_not_calibrated=True, excluded_train_images=len(rows) - len(eligible))
    if not np.allclose(score_features(model, x), classifier.predict_proba(scaler.transform(x))[:, 1], atol=1e-12, rtol=0):
        raise RuntimeError('Portable JSON selector replay mismatch')
    return model


def decisions(rows, scores, threshold=.5):
    """Labels/outcomes/classes are not consulted for ANY selection decision."""
    if len(rows) != len(scores):
        raise ValueError('Score/manifest length mismatch')
    eligible = [i for i, r in enumerate(rows) if r['geometry_valid']]
    chosen = {i for i in eligible if scores[i] >= threshold}
    result = dict(baseline=set(), all_single=set(eligible), selector=chosen)
    # Match the selector's number of accepted images, never its label/class mix.
    result['confidence_matched'] = set(sorted(eligible, key=lambda i: (-rows[i]['features'][0], rows[i]['image']))[:len(chosen)])
    canonical = sorted(eligible, key=lambda i: rows[i]['image'])
    for seed in SEEDS:
        positions = np.random.RandomState(seed).permutation(len(canonical))[:len(chosen)]
        result['random_%d' % seed] = {canonical[int(j)] for j in positions}
    return result


def stage_split(path, report, scratch, threshold):
    """Replay baseline/A only. Ignore medical rows for this industrial pilot."""
    side = report['metadata']['image_size']
    selected = [r for r in report['per_image'] if r['source'] == 'mvtec']
    if not selected:
        raise ValueError('No MVTec source images')
    scratch.mkdir()
    rows, classes, max_delta = [], [], 0.
    for class_index, category in enumerate(sorted({r['class'] for r in selected})):
        originals = [r for r in selected if r['class'] == category]
        pools = {m: np.lib.format.open_memmap(scratch / ('class_%02d_%s.npy' % (class_index, m)),
            mode='w+', dtype=np.bool_ if m == 'mask' else np.float32,
            shape=(len(originals), side, side)) for m in ('baseline', 'all_single', 'mask')}
        indices = []
        for position, original in enumerate(originals):
            with np.load(safe_child(path.parent, original['cache']), allow_pickle=False) as cache:
                base, crop, support, shared, mask = (cache[k] for k in
                    ('baseline', 'crop_a', 'update_support', 'shared_support', 'gt_mask'))
            if (any(v.shape != (side, side) for v in (base, crop, support, shared, mask))
                    or any(v.dtype != np.bool_ for v in (support, shared, mask))
                    or base.dtype != np.float32 or crop.dtype != np.float32):
                raise ValueError('Invalid source cache shape/dtype')
            valid = original['evidence']['valid']
            if type(valid) is not bool or original['anomaly'] not in (0, 1):
                raise ValueError('Invalid source geometry/label')
            if not np.array_equal(support, shared if valid else np.zeros_like(shared)):
                raise ValueError('Source support differs from original geometry policy')
            for values in (base, crop):
                if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
                    raise ValueError('Invalid source probabilities')
            if valid and not support.any():
                raise ValueError('Valid geometry requires nonempty support')
            single = base.copy()
            single[support] = .5 * base[support] + .5 * crop[support]
            base, single = gaussian_filter(base, sigma=4), gaussian_filter(single, sigma=4)
            features = prediction_features(base, single, support)  # GT enters only BELOW.
            if original['anomaly'] == 0 and mask.any():
                raise ValueError('Normal source image has positive GT pixels')
            calculated = dict(baseline=metrics(mask, base), all_single=metrics(mask, single))
            for m, old in (('baseline', 'baseline'), ('all_single', 'single_crop_fixed_fusion')):
                max_delta = max(max_delta, check_replay(original['metrics'][old], calculated[m]))
            before, after = calculated['baseline']['pixel_ap'], calculated['all_single']['pixel_ap']
            ap_delta = after - before if before is not None and after is not None else None
            counts = dict(baseline=pixel_counts(mask, base, threshold), all_single=pixel_counts(mask, single, threshold))
            fp_delta = counts['all_single']['fp'] - counts['baseline']['fp']
            indices.append(len(rows))
            rows.append(dict(image=original['image'], **{'class': category}, anomaly=original['anomaly'],
                geometry_valid=valid, features=features, metrics=calculated, counts=counts, ap_delta=ap_delta,
                normal_fp_delta=fp_delta if original['anomaly'] == 0 else None,
                utility_label=utility_label(original['anomaly'], ap_delta, fp_delta), cache_position=position))
            pools['baseline'][position], pools['all_single'][position], pools['mask'][position] = base, single, mask
        for pool in pools.values():
            pool.flush()
        del pool, pools
        classes.append(dict(name=category, cache_index=class_index, row_indices=indices))
        print('%s staged %s images=%d replay_max_delta=%g' % (
            report['metadata'].get('split', 'validation'), category, len(originals), max_delta), flush=True)
    return rows, classes, max_delta


def association(rows, scores, anomaly=None):
    indices = [i for i, r in enumerate(rows) if r['geometry_valid'] and r['utility_label'] is not None
               and (anomaly is None or r['anomaly'] == anomaly)]
    y = [rows[i]['utility_label'] for i in indices]
    return dict(defined_images=len(indices), beneficial=sum(y), harmful=len(y) - sum(y),
        auroc_beneficial_vs_harmful=float(roc_auc_score(y, [scores[i] for i in indices])) if set(y) == {0, 1} else None)


def summarize_policy(rows, accepted):
    total = empty_counts()
    for i, row in enumerate(rows):
        accumulate(total, row['counts']['all_single' if i in accepted else 'baseline'], row['anomaly'] == 0)
    classes = sorted({r['class'] for r in rows})
    by_class_ap, gains = [], []
    for category in classes:
        defined = [(i, r) for i, r in enumerate(rows) if r['class'] == category and r['anomaly'] == 1 and r['ap_delta'] is not None]
        if defined:
            by_class_ap.append(mean([r['metrics']['all_single' if i in accepted else 'baseline']['pixel_ap'] for i, r in defined]))
            gains.append(mean([r['ap_delta'] if i in accepted else 0. for i, r in defined]))
    return dict(accepted_images=len(accepted), images=len(rows),
        accepted_normal=sum(rows[i]['anomaly'] == 0 for i in accepted),
        accepted_anomalous=sum(rows[i]['anomaly'] == 1 for i in accepted),
        accepted_beneficial=sum(rows[i]['utility_label'] == 1 for i in accepted),
        accepted_harmful=sum(rows[i]['utility_label'] == 0 for i in accepted),
        accepted_harmful_anomalous=sum(rows[i]['utility_label'] == 0 and rows[i]['anomaly'] == 1 for i in accepted),
        accepted_harmful_normal=sum(rows[i]['utility_label'] == 0 and rows[i]['anomaly'] == 0 for i in accepted),
        rejected_beneficial=sum(r['utility_label'] == 1 and i not in accepted for i, r in enumerate(rows)),
        rejected_harmful=sum(r['utility_label'] == 0 and i not in accepted for i, r in enumerate(rows)),
        class_balanced_anomalous_image_ap=mean(by_class_ap), class_balanced_anomalous_image_ap_gain=mean(gains),
        operating_point=finish(total))


def evaluate_split(rows, classes, scratch, model):
    scores = score_features(model, [r['features'] for r in rows])
    choices = decisions(rows, scores, model['acceptance_threshold'])
    policies = {p: summarize_policy(rows, choices[p]) for p in POLICIES}
    class_ap = {}
    for group in classes:
        index = group['cache_index']
        cache = {m: np.load(scratch / ('class_%02d_%s.npy' % (index, m)), mmap_mode='r')
                 for m in ('baseline', 'all_single', 'mask')}
        mixed = np.lib.format.open_memmap(scratch / 'mixture.npy', mode='w+', dtype=np.float32, shape=cache['mask'].shape)
        ap = {}
        for p in POLICIES:
            if p in ('baseline', 'all_single'):
                values = cache[p]
            else:
                for position, row_index in enumerate(group['row_indices']):
                    mixed[position] = cache['all_single' if row_index in choices[p] else 'baseline'][position]
                mixed.flush()
                values = mixed
            ap[p] = pooled_ap(cache['mask'], values)
        class_ap[group['name']] = ap
        del values, mixed, cache
        print('Pooled AP evaluated: ' + group['name'], flush=True)
    for p in POLICIES:
        policies[p]['selected_subset_class_pooled_ap_percent'] = mean([v[p] for v in class_ap.values()])
    probability = len(choices['selector']) / sum(r['geometry_valid'] for r in rows) if any(r['geometry_valid'] for r in rows) else 0.
    expected_gains = []
    for category in sorted({r['class'] for r in rows}):
        deltas = [probability * r['ap_delta'] if r['geometry_valid'] else 0. for r in rows
                  if r['class'] == category and r['anomaly'] == 1 and r['ap_delta'] is not None]
        if deltas:
            expected_gains.append(mean(deltas))
    random_ap = [policies['random_%d' % s]['selected_subset_class_pooled_ap_percent'] for s in SEEDS]
    for i, row in enumerate(rows):
        row['selector_score'] = float(scores[i])
        row['decisions'] = {p: i in choices[p] for p in POLICIES}
    return dict(images=len(rows), valid_images=sum(r['geometry_valid'] for r in rows),
        anomalous_images_without_defined_ap=sum(r['anomaly'] == 1 and r['ap_delta'] is None for r in rows),
        associations=dict(combined_proxy=association(rows, scores), anomalous_ap=association(rows, scores, 1),
                          normal_fp=association(rows, scores, 0)),
        policies=policies, by_class_pooled_ap_percent=class_ap,
        matched_random=dict(seeds=list(SEEDS), accepted_per_seed=len(choices['selector']),
            pooled_ap_mean_percent=mean(random_ap), pooled_ap_std_percent=float(np.std(random_ap)),
            pooled_ap_min_percent=min(random_ap), pooled_ap_max_percent=max(random_ap),
            exact_expected_class_balanced_anomalous_image_ap_gain=mean(expected_gains),
            note='Each seed selects exactly k valid images uniformly without replacement; mean of separately computed pooled APs, NOT an averaged map or exact expected pooled AP. Five-seed spread is not a confidence interval.'),
        comparisons=dict(selector_vs_all_single_pooled_ap_pp=policies['selector']['selected_subset_class_pooled_ap_percent'] - policies['all_single']['selected_subset_class_pooled_ap_percent'],
            selector_vs_random_mean_pooled_ap_pp=policies['selector']['selected_subset_class_pooled_ap_percent'] - mean(random_ap),
            selector_vs_confidence_pooled_ap_pp=policies['selector']['selected_subset_class_pooled_ap_percent'] - policies['confidence_matched']['selected_subset_class_pooled_ap_percent']))


def main(args):
    # Existing provenance guard checks exact source train/validation identities,
    # matching configurations, frozen/equivalence checks and path isolation.
    paths, reports, output, reference = prepare(args)
    output.mkdir(parents=True, exist_ok=True)
    operating = calibrate_pixel_threshold(reports[0], paths[0].parent, output, .001)
    # This helper reads RAW source caches; correct its target-cache-specific note.
    operating.pop('target_cache_already_filtered_no_second_smoothing')
    operating['source_candidate_smoothing'] = 'sigma4 once after raw 0.5 fusion; never filter staged maps again'
    result = dict(status='running', protocol=dict(no_model_inference=True, no_target_data=True,
        clip_or_prompts_modified=False, selector_training=True, train_source='mvtec only; colondb excluded',
        selector_fit_labels_from_train_only=True, validation_labels_used_for_evaluation_only=True,
        no_validation_rule_selection=True, no_hyperparameter_search=True, acceptance_threshold=.5,
        feature_names=list(FEATURES), single_view='crop_a only, original shared trimmed support; no crop_b features',
        fusion_weight=.5, source_pixel_fpr=.001, concentration='largest ceil(0.1*N) support values / support sum; zero if sum=0',
        constant_rank_correlation=0., invalid_geometry='preserve baseline; exclude from fit and matched-budget pool',
        metrics='subset category-pooled AP including normals, then unweighted category mean; per-image AP is separate',
        caveats='Exploratory source validation has been reused, and source classes were seen by the frozen AdaCLIP. '
            'This is not an independent unseen-target test. Proxy labels mix anomalous AP gain with normal fixed-threshold FP reduction; '
            'normal FP reduction need not improve pooled AP. Predictions require the crop result, so this does not avoid crop inference.'),
        operating_point=operating, calibration_report_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        validation_report_sha256=hashlib.sha256(paths[1].read_bytes()).hexdigest(),
        reference_report_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(), splits={})
    summary_path = output / 'source_single_utility_summary.json'
    summary_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    with tempfile.TemporaryDirectory(prefix='single_selector_scratch_', dir=output) as temporary:
        scratch = Path(temporary)
        train, train_classes, replay = stage_split(paths[0], reports[0], scratch / 'train', operating['pixel_threshold'])
        model = fit_selector(train)  # Validation has not even been staged yet.
        model.update(status='fitted_source_only', source_config={k: reports[0]['metadata'][k] for k in CONFIG_KEYS},
            calibration_report_sha256=result['calibration_report_sha256'], operating_point=operating,
            intended_use='SOURCE PILOT ONLY; no automatic target evaluation or model insertion')
        (output / 'source_single_utility_selector.json').write_text(json.dumps(model, indent=2, allow_nan=False), encoding='utf8')
        for split, path, report in zip(('train', 'validation'), paths, reports):
            if split == 'train':
                rows, classes = train, train_classes
            else:
                rows, classes, replay = stage_split(path, report, scratch / split, operating['pixel_threshold'])
            result['splits'][split] = evaluate_split(rows, classes, scratch / split, model)
            result['splits'][split].update(replay_passed=True, replay_max_absolute_metric_delta=replay,
                ignored_colondb_images=sum(r['source'] != 'mvtec' for r in report['per_image']))
            (output / (split + '_single_utility_images.json')).write_text(json.dumps(rows, indent=2, allow_nan=False), encoding='utf8')
            summary_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    result['status'] = 'complete'
    summary_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    lines = ['Source single-A utility selector DONE',
        'ONE fixed source-train logistic model; threshold=0.5, no validation/target tuning.',
        'Source-subset results only, NOT full-target benchmarks or a latency-saving method.',
        'Features=' + ', '.join(FEATURES)]
    for split, entry in result['splits'].items():
        lines.append('%s images=%d associations=%s' % (split, entry['images'], json.dumps(entry['associations'])))
        for p in ('baseline', 'all_single', 'selector', 'confidence_matched'):
            d = entry['policies'][p]
            lines.append('  %s accepted=%d P-AP=%.6f image_AP_gain=%s normal_pixel_FPR=%s recall=%s harmful_accepted=%d' % (
                p, d['accepted_images'], d['selected_subset_class_pooled_ap_percent'], d['class_balanced_anomalous_image_ap_gain'],
                d['operating_point']['normal_pixel_fpr'], d['operating_point']['pixel_recall'], d['accepted_harmful']))
        lines.append('  matched_random=' + json.dumps(entry['matched_random']))
        lines.append('  comparisons=' + json.dumps(entry['comparisons']))
    text = '\n'.join(lines) + '\n'
    (output / 'source_single_utility_summary.txt').write_text(text, encoding='utf8')
    print(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration_report', required=True)
    parser.add_argument('--validation_report', required=True)
    parser.add_argument('--save_path', required=True)
    main(parser.parse_args())
