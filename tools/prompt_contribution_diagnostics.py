"""Opt-in, source-only prompt Value intervention. Compatible with Torch 1.10.

Hooks exist only inside this diagnostic. No backbone/attention implementation or
checkpoint format is changed. The correction uses projected V (including its
bias), keeps Q/K/softmax unchanged at the current input, and has no output bias.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import statistics

import torch
from torch.nn import functional as F

from .gate_equivalence import _rng_state, _restore_rng, _prompt_pair_metrics


MODES = ('baseline', 'contribution_1', 'amplitude_shared_0.5',
         'amplitude_visual_0.5', 'contribution_0.5')
LAYERWISE_MODES = ('baseline', 'contribution_1') + tuple(
    f'{kind}_visual_L{layer}_0.5'
    for layer in range(1, 5) for kind in ('amplitude', 'contribution'))
METRICS = ('classification', 'segmentation', 'total')
AP_TOL = 1e-4


def report_name(args):
    return ('source_prompt_layer_sensitivity' if
            getattr(args, 'prompt_contribution_scope', 'all') == 'layerwise'
            else 'source_prompt_contribution')


def intervention_spec(mode):
    """Return intervention family and zero-based target; None means all layers."""
    if mode not in MODES + LAYERWISE_MODES:
        raise ValueError('Unknown prompt intervention: ' + mode)
    layer = int(mode.split('_L')[1].split('_')[0]) - 1 if '_L' in mode else None
    return mode.split('_')[0], layer


def validate_args(args):
    if (not args.load_baseline or args.fusion_mode != 'shared_gate'
            or args.prompting_type != 'SD' or args.prompting_branch != 'VL'
            or args.prompting_depth != 4 or args.batch_size != 1
            or args.testing_model != 'dataset' or args.gate_override is not None
            or args.check_gate_equivalence or args.diagnose_prompt_scale
            or args.diagnose_source_losses or args.compare_loss_shapes
            or args.source_reference_report):
        raise ValueError('Contribution diagnosis requires baseline import, shared_gate, SD/VL, depth 4, '
                         'batch 1, dataset mode and no other diagnostic/override')
    if not args.contribution_reference_report:
        raise ValueError('Supply the previous direct_gate_pilot.json with --contribution_reference_report')
    if not Path(args.contribution_reference_report).is_file():
        raise FileNotFoundError(args.contribution_reference_report)
    scope = getattr(args, 'prompt_contribution_scope', 'all')
    threshold = getattr(args, 'contribution_pixel_threshold', None)
    if scope not in ('all', 'layerwise'):
        raise ValueError('Unknown prompt_contribution_scope')
    if scope == 'layerwise':
        if threshold is None or not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError('Layerwise diagnosis requires a fixed --contribution_pixel_threshold in (0,1)')
    elif threshold is not None:
        raise ValueError('--contribution_pixel_threshold requires layerwise scope')
    if (Path(args.save_path) / 'diagnostics' / (report_name(args) + '.json')).exists():
        raise FileExistsError('Use a new --save_path; an existing report will not be overwritten')


def aggregate_detection_map(patch_tokens, text_features, image_size):
    """Native test-time order: interpolate raw logits, mean layers, then softmax.

    Do not average the already-softmaxed training maps or recover clipped logits
    from probabilities. Reuses the current forward's normalized features, without
    a second encoder or HSF call.
    """
    logits = []
    for patch in patch_tokens:
        value = 100.0 * patch @ text_features
        batch, tokens, channels = value.shape
        side = math.isqrt(tokens)
        if side * side != tokens or channels != 2:
            raise ValueError('Expected square patch grid with two text states')
        value = value.permute(0, 2, 1).view(batch, 2, side, side)
        logits.append(F.interpolate(value, size=image_size, mode='bilinear', align_corners=True))
    probs = torch.softmax(torch.mean(torch.stack(logits, dim=1), dim=1), dim=1)
    return (probs[:, 1:] + 1 - probs[:, :1]) / 2.0


def prompt_value_contribution(attn, inputs, length, chunk_size=128):
    """Return output-projected prompt contribution for NON-prompt queries only.

Only the exact self-attention configuration used by the visual CLIP blocks is
supported. Chunking avoids a second full [heads, tokens, tokens] allocation.
"""
    q, k, v = inputs[:3]
    if (q is not k or q is not v or q.ndim != 3 or q.shape[1] != 1
            or not 0 < length < q.shape[0] or chunk_size < 1
            or attn.training or attn.dropout != 0
            or getattr(attn, 'batch_first', False) or not attn._qkv_same_embed_dim
            or attn.bias_k is not None or attn.bias_v is not None or attn.add_zero_attn):
        raise ValueError('Unsupported attention configuration for prompt contribution diagnosis')
    tokens, batch, width = q.shape
    heads, head_dim = attn.num_heads, width // attn.num_heads
    # Fused projection follows the self-attention path in Torch 1.10 MHA.
    pq, pk, pv = F.linear(q, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
    def split(x):
        return x.contiguous().view(tokens, batch * heads, head_dim).transpose(0, 1)
    pq, pk, pv = split(pq) * (head_dim ** -0.5), split(pk), split(pv)
    output = []
    for start in range(0, tokens - length, chunk_size):
        stop = min(start + chunk_size, tokens - length)
        weights = torch.bmm(pq[:, start:stop], pk.transpose(1, 2)).softmax(dim=-1)
        message = torch.bmm(weights[:, :, -length:], pv[:, -length:])
        message = message.transpose(0, 1).contiguous().view(stop - start, batch, width)
        output.append(F.linear(message, attn.out_proj.weight, None))
    result = torch.cat(output)
    if not torch.isfinite(result).all():
        raise RuntimeError('Non-finite prompt Value contribution')
    return result


@contextmanager
def capture_intervention(model, mode, capture_detection=False):
    """Do not intercept the initial, unprompted image-conditioning forward."""
    family, selected_layer = intervention_spec(mode)
    p = model.visual_prompter
    blocks = model.visual.transformer.resblocks
    captures, handles = {}, []
    active = [None]
    old_gates = [None]
    prompters = (p, model.text_prompter)
    original_modes = [model.fusion_mode] + [x.fusion_mode for x in prompters]
    generator = model.shared_gate_generator
    similarity_was_local = 'visual_text_similarity' in model.__dict__
    original_similarity = model.visual_text_similarity if capture_detection else None

    def similarity(image_feature, patch_token, text_feature, aggregation):
        result = original_similarity(image_feature, patch_token, text_feature, aggregation)
        if 'prediction_map' in captures:
            raise RuntimeError('Repeated detection-map capture')
        captures['prediction_map'] = aggregate_detection_map(
            patch_token, text_feature, model.image_size).detach().float().cpu()
        return result

    def before_prompt(module, inputs):
        block, index = inputs[:2]
        if active[0] is not None:
            raise RuntimeError('Unexpected nested visual prompter')
        active[0] = index
        if family == 'amplitude' and mode != 'amplitude_shared_0.5' and (
                selected_layer is None or index == selected_layer):
            old_gates[0] = module.dynamic_gates
            if selected_layer is None:
                module.dynamic_gates = torch.full_like(module.dynamic_gates, 0.5)
            else:
                module.dynamic_gates = module.dynamic_gates.clone()
                module.dynamic_gates[:, selected_layer] = 0.5

    def after_prompt(module, inputs, output):
        if old_gates[0] is not None:
            module.dynamic_gates = old_gates[0]
            old_gates[0] = None
        active[0] = None

    def attention_hook(index):
        def hook(module, inputs, output):
            if active[0] != index:
                return None
            key = 'attention_visual_' + str(index)
            if key in captures:
                raise RuntimeError('Repeated prompted attention capture')
            value, weights = output
            changed = value
            if family == 'contribution' and (selected_layer is None or index == selected_layer):
                contribution = prompt_value_contribution(module, inputs, p.length)
                captures['message_visual_' + str(index)] = contribution.detach().float().cpu()
                if mode != 'contribution_1':
                    changed = value.clone()
                    changed[:-p.length] = value[:-p.length] - 0.5 * contribution
            captures[key] = changed[:-p.length].detach().float().cpu().clone()
            return changed, weights
        return hook

    def feature_hook(label):
        def hook(module, inputs, output):
            for i, value in enumerate(output):
                key = label + '_' + str(i)
                if key in captures:
                    raise RuntimeError('Repeated projected feature capture')
                captures[key] = value.detach().float().cpu().clone()
        return hook

    try:
        if capture_detection:
            model.visual_text_similarity = similarity
        if mode == 'baseline':
            model.fusion_mode = 'add'
            model.shared_gate_generator = None
            for prompter in prompters:
                prompter.fusion_mode = 'add'
        handles.append(p.register_forward_pre_hook(before_prompt))
        handles.append(p.register_forward_hook(after_prompt))
        for i in range(p.depth):
            handles.append(blocks[i].attn.register_forward_hook(attention_hook(i)))
        handles.append(model.cls_token_layer.register_forward_hook(feature_hook('projected_cls')))
        handles.append(model.patch_token_layer.register_forward_hook(feature_hook('projected_patch')))
        yield captures
        required = {'attention_visual_' + str(i) for i in range(p.depth)}
        required |= {'projected_cls_0'} | {'projected_patch_' + str(i) for i in range(len(model.output_layers))}
        if capture_detection:
            required.add('prediction_map')
        if not required <= set(captures) or active[0] is not None:
            raise RuntimeError('Incomplete prompted-path captures')
    finally:
        for handle in handles:
            handle.remove()
        if capture_detection:
            if similarity_was_local:
                model.visual_text_similarity = original_similarity
            else:
                delattr(model, 'visual_text_similarity')
        if old_gates[0] is not None:
            p.dynamic_gates = old_gates[0]
        model.shared_gate_generator = generator
        model.fusion_mode = original_modes[0]
        for prompter, original in zip(prompters, original_modes[1:]):
            prompter.fusion_mode = original


def source_loaders(trainer, args, get_data):
    """Resolve exact recorded pilot validation identities, never resample."""
    with open(args.contribution_reference_report, encoding='utf8') as f:
        reference = json.load(f)
    meta = reference['metadata']
    for key in ('model', 'image_size', 'use_hsf', 'k_clusters', 'prompting_type',
                'prompting_branch', 'prompting_depth', 'prompting_length'):
        if meta.get(key) != getattr(args, key):
            raise ValueError('Pilot configuration mismatch: ' + key)
    if Path(meta['baseline_checkpoint']).resolve() != Path(args.ckt_path).resolve():
        raise ValueError('Use the original baseline checkpoint recorded in the pilot report')
    sources = meta['sources']
    if not sources or len(set(sources)) != len(sources) or not set(sources) <= {'mvtec', 'colondb'}:
        raise ValueError('Only distinct mvtec/colondb source datasets are allowed')
    rows = reference['validation']['per_image']
    if not rows or reference['validation']['overall']['images'] != len(rows):
        raise ValueError('Invalid pilot validation records')
    if {row['source'] for row in rows} != set(sources):
        raise ValueError('Pilot validation source coverage mismatch')
    loaders, seen = [], set()
    for source in sources:
        _, dataset, _ = get_data(dataset_type_list=source, transform=trainer.preprocess,
                                target_transform=trainer.transform, training=False)
        if dataset.training:
            raise ValueError('Augmentation must be off')
        mapping = {}
        for i, item in enumerate(dataset.data_all):
            path = str((Path(dataset.root) / item['img_path']).resolve())
            if path in mapping:
                raise ValueError('Duplicate dataset image path: ' + path)
            mapping[path] = (i, item)
        indices = []
        for row in (r for r in rows if r['source'] == source):
            path = str(Path(row['image']).resolve())
            if path in seen or path not in mapping:
                raise ValueError('Duplicate or missing pilot image: ' + path)
            seen.add(path)
            i, item = mapping[path]
            if item['cls_name'] != row['class'] or int(item['anomaly']) != row['anomaly']:
                raise ValueError('Pilot class/label mismatch: ' + path)
            indices.append(i)
        loaders.append((source, torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, indices), batch_size=1, shuffle=False)))
    return loaders, {'reference_report': str(Path(args.contribution_reference_report).resolve()),
                     'baseline_checkpoint': str(Path(args.ckt_path).resolve()),
                     'pilot_metadata': meta, 'images': len(rows)}


def mask_information(items):
    import numpy as np
    mask = items['img_mask'].detach().cpu().numpy()
    if mask.ndim != 4 or mask.shape[:2] != (1, 1) or not np.isfinite(mask).all():
        raise ValueError('Expected one finite [1,1,H,W] source mask')
    mask = mask[0, 0] > 0.5
    fraction = float(mask.mean())
    normal = int(items['anomaly'][0]) == 0
    if normal and mask.any():
        raise ValueError('Normal source image has foreground mask pixels')
    group = ('normal' if normal else 'empty_anomaly_mask' if fraction == 0
             else 'small' if fraction <= .01 else 'medium' if fraction <= .05 else 'large')
    return mask, fraction, group


def pixel_measurements(prediction, mask, threshold):
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from sklearn.metrics import average_precision_score
    values = prediction.detach().float().cpu().numpy()
    if values.shape != (1, 1) + mask.shape or not np.isfinite(values).all():
        raise ValueError('Detection map/mask shape mismatch or non-finite prediction')
    if values.min() < 0 or values.max() > 1:
        raise ValueError('Expected detection probabilities in [0,1]')
    values = gaussian_filter(values[0, 0], sigma=4)
    positive = values > threshold
    foreground, background = int(mask.sum()), int((~mask).sum())
    tp, fp = int((positive & mask).sum()), int((positive & ~mask).sum())
    return {'pixel_ap': float(average_precision_score(mask.ravel(), values.ravel())) if foreground else None,
            'true_positive_pixels': tp, 'false_positive_pixels': fp,
            'foreground_pixels': foreground, 'background_pixels': background,
            'pixel_recall': tp / foreground if foreground else None,
            'background_pixel_fpr': fp / background if background else None,
            'score_mean': float(values.mean()), 'score_q99': float(np.quantile(values, .99)),
            'score_max': float(values.max())}


def evaluation_summary(rows, modes):
    result = {}
    normals = [r for r in rows if r['anomaly'] == 0]
    for mode in modes:
        valid = [r for r in rows if r['anomaly'] == 1 and
                 r['evaluation'][mode]['pixel_ap'] is not None]
        deltas = [r['evaluation'][mode]['pixel_ap'] - r['evaluation']['baseline']['pixel_ap']
                  for r in valid]
        pixels = sum(r['evaluation'][mode]['background_pixels'] for r in normals)
        fp = sum(r['evaluation'][mode]['false_positive_pixels'] for r in normals)
        fp_deltas = [r['evaluation'][mode]['false_positive_pixels'] -
                     r['evaluation']['baseline']['false_positive_pixels'] for r in normals]
        result[mode] = {
            'anomalous_ap_defined_images': len(valid),
            'anomalous_ap_undefined_images': sum(r['anomaly'] == 1 for r in rows) - len(valid),
            'mean_anomalous_image_pixel_ap': statistics.mean(r['evaluation'][mode]['pixel_ap']
                                                           for r in valid) if valid else None,
            'mean_image_ap_delta': statistics.mean(deltas) if deltas else None,
            'image_ap_changes': {'improves': sum(d > AP_TOL for d in deltas),
                                 'worsens': sum(d < -AP_TOL for d in deltas),
                                 'ties': sum(abs(d) <= AP_TOL for d in deltas)},
            'normal_images': len(normals), 'normal_pixels': pixels, 'normal_fp_pixels': fp,
            'normal_pixel_fpr': fp / pixels if pixels else None,
            'normal_fp_changes': {'decreases': sum(d < 0 for d in fp_deltas),
                                  'increases': sum(d > 0 for d in fp_deltas),
                                  'ties': sum(d == 0 for d in fp_deltas)}}
    return result


def layer_preference_summary(rows):
    """Retrospective GT-based preferences only; never feed them into the model."""
    result = {}
    for family in ('amplitude', 'contribution'):
        counts = {str(i): 0 for i in range(1, 5)}
        unique = counts.copy()
        beneficial, no_gain, mixed = 0, 0, 0
        for row in rows:
            if row['anomaly'] != 1 or row['evaluation']['baseline']['pixel_ap'] is None:
                continue
            base = row['evaluation']['baseline']['pixel_ap']
            deltas = {i: row['evaluation'][f'{family}_visual_L{i}_0.5']['pixel_ap'] - base
                      for i in range(1, 5)}
            best = max(deltas.values())
            mixed += int(any(d > AP_TOL for d in deltas.values()) and
                         any(d < -AP_TOL for d in deltas.values()))
            if best <= AP_TOL:
                no_gain += 1
                continue
            beneficial += 1
            winners = [i for i, delta in deltas.items() if abs(delta - best) <= AP_TOL]
            for i in winners:
                counts[str(i)] += 1
            if len(winners) == 1:
                unique[str(winners[0])] += 1
        result[family] = {'images_with_beneficial_layer': beneficial,
                          'images_without_beneficial_layer': no_gain,
                          'images_with_both_helpful_and_harmful_layers': mixed,
                          'tie_inclusive_best_beneficial_layer_counts': counts,
                          'unique_best_beneficial_layer_counts': unique}
    return result


def summarize(rows, modes=MODES):
    means = {mode: {metric: statistics.mean(r['losses'][mode][metric] for r in rows)
                    for metric in METRICS} for mode in modes}
    comparisons = {}
    for mode in modes[2:]:
        comparisons[mode] = {}
        for metric in METRICS:
            deltas = [r['losses'][mode][metric] - r['losses']['baseline'][metric] for r in rows]
            ties = [math.isclose(r['losses'][mode][metric], r['losses']['baseline'][metric],
                                 abs_tol=1e-6, rel_tol=1e-5) for r in rows]
            comparisons[mode][metric] = {'mean_delta': statistics.mean(deltas),
                'improves': sum(d < 0 and not t for d, t in zip(deltas, ties)),
                'worsens': sum(d > 0 and not t for d, t in zip(deltas, ties)), 'ties': sum(ties)}
    changes = {}
    for mode in modes[2:]:
        changes[mode] = {}
        for key in rows[0]['changes'][mode]:
            values = [r['changes'][mode][key]['relative_l2'] for r in rows]
            valid = [v for v in values if v is not None]
            changes[mode][key] = {'mean_relative_l2': statistics.mean(valid) if valid else None,
                                  'valid_count': len(valid)}
    result = {'images': len(rows), 'mean_losses': means, 'vs_baseline': comparisons,
              'representation_changes': changes}
    if 'evaluation' in rows[0]:
        result['detection_metrics'] = evaluation_summary(rows, modes)
        result['layer_preferences_gt_diagnostic_only'] = layer_preference_summary(rows)
    return result


def diagnose(trainer, loaders, logger, output_path, metadata, layerwise=False, pixel_threshold=None):
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(path)
    if layerwise and (pixel_threshold is None or not math.isfinite(pixel_threshold)
                      or not 0 < pixel_threshold < 1):
        raise ValueError('Layerwise diagnosis requires a fixed pixel threshold in (0,1)')
    modes = LAYERWISE_MODES if layerwise else MODES
    model = trainer.clip_model
    if (trainer.fusion_mode != 'shared_gate' or model.static_only
            or not model.visual_prompter.enabled or model.visual_prompter.depth != 4):
        raise ValueError('Expected imported baseline with four active visual prompt layers')
    flags = [(m, m.training) for m in model.modules()]
    grad_flags = [(p, p.requires_grad) for p in model.parameters()]
    versions = {k: v._version for k, v in model.state_dict().items()}
    before = {k: v.detach().cpu().clone() for k, v in trainer.state_dict().items()
              if any(part in k for part in trainer.learnable_paramter_list)}
    rng = _rng_state()
    rows, maximum = [], 0.0
    try:
        model.eval()
        for p, _ in grad_flags:
            p.requires_grad_(False)
        with torch.no_grad():
            for source, loader in loaders:
                for items in loader:
                    if items['img'].shape[0] != 1:
                        raise ValueError('Batch size must be one')
                    paired_rng = _rng_state()
                    losses, changes, messages, evaluation = {}, {}, {}, {}
                    if layerwise:
                        mask, area_fraction, area_group = mask_information(items)
                    readout_metrics = {}
                    reference_capture, reference_output = None, None
                    for mode in modes:
                        _restore_rng(paired_rng)
                        gate = None if mode == 'baseline' else (0.5 if mode == 'amplitude_shared_0.5' else 1.0)
                        with capture_intervention(model, mode, capture_detection=layerwise) as captures:
                            maps, scores = model(items['img'].to(trainer.device), items['cls_name'],
                                                 aggregation=False, gate_override=gate)
                        outputs = [m.detach().float().cpu() for m in maps] + [scores.detach().float().cpu()]
                        if any(not torch.isfinite(x).all() for x in outputs):
                            raise RuntimeError('Non-finite model output')
                        if layerwise:
                            evaluation[mode] = pixel_measurements(captures['prediction_map'], mask, pixel_threshold)
                            if mode == 'baseline':
                                readout_metrics = {str(layer): pixel_measurements(
                                    (m[:, 1:] + 1 - m[:, :1]) / 2.0, mask, pixel_threshold)
                                    for layer, m in zip(model.output_layers, maps)}
                        targets = {**items, 'img_mask': items['img_mask'].clone(),
                                   'anomaly': items['anomaly'].clone()}
                        components = trainer.detection_loss(maps, scores, targets,
                                                            return_components=True, preserve_batch_dim=True)
                        losses[mode] = {m: components[m].item() for m in METRICS}
                        if not all(math.isfinite(v) for v in losses[mode].values()):
                            raise RuntimeError('Non-finite diagnostic loss')
                        if not math.isclose(losses[mode]['total'], losses[mode]['classification'] +
                                            losses[mode]['segmentation'], rel_tol=1e-5, abs_tol=1e-6):
                            raise RuntimeError('Loss components do not sum to total')
                        if mode == 'baseline':
                            reference_capture, reference_output = captures, outputs
                        elif mode == 'contribution_1':
                            if len(reference_output) != len(outputs):
                                raise RuntimeError('Contribution=1 output count mismatch')
                            for a, b in zip(reference_output, outputs):
                                if a.shape != b.shape or not torch.allclose(a, b, atol=1e-6, rtol=1e-5):
                                    raise RuntimeError('Contribution=1 baseline equivalence FAILED')
                                maximum = max(maximum, (a - b).abs().max().item())
                            for key, a in reference_capture.items():
                                if not torch.allclose(a, captures[key], atol=1e-6, rtol=1e-5):
                                    raise RuntimeError('Contribution=1 feature equivalence FAILED: ' + key)
                        else:
                            changes[mode] = {key: _prompt_pair_metrics(a, captures[key])
                                             for key, a in reference_capture.items()}
                            changes[mode]['score'] = _prompt_pair_metrics(reference_output[-1], outputs[-1])
                            for i, (a, b) in enumerate(zip(reference_output[:-1], outputs[:-1])):
                                changes[mode]['map_' + str(i)] = _prompt_pair_metrics(a, b)
                        if mode.startswith('contribution_'):
                            messages[mode] = {}
                            for i in range(model.visual_prompter.depth):
                                if 'message_visual_' + str(i) not in captures:
                                    continue
                                message = captures['message_visual_' + str(i)]
                                out = captures['attention_visual_' + str(i)]
                                # Reconstruct the local, ungated output at this SAME layer input.
                                original = out + (0.5 * message if mode != 'contribution_1' else 0)
                                denominator = original.norm().item()
                                messages[mode][str(i)] = {'norm': message.norm().item(),
                                    'relative_to_local_attention': message.norm().item() / denominator
                                    if denominator > 1e-12 else None}
                    row = {'source': source, 'class': items['cls_name'][0],
                           'anomaly': int(items['anomaly'][0]), 'image': items['img_path'][0],
                           'losses': losses, 'changes': changes, 'prompt_messages': messages}
                    if layerwise:
                        row.update(anomaly_area_fraction=area_fraction, area_group=area_group,
                                   evaluation=evaluation, baseline_readout_metrics=readout_metrics)
                    rows.append(row)
                    logger.info(f'Contribution image={len(rows)} source={source} class={row["class"]} '
                                f'anomaly={row["anomaly"]} equivalence=PASS totals=' +
                                json.dumps({m: losses[m]['total'] for m in modes}))
        if not rows or len(rows) != metadata['images']:
            raise RuntimeError('Reference image count mismatch')
        current = trainer.state_dict()
        if any(not torch.equal(v, current[k].detach().cpu()) for k, v in before.items()):
            raise RuntimeError('Saved detector/gate tensors changed')
        if any(v._version != versions[k] for k, v in model.state_dict().items()):
            raise RuntimeError('Model tensor version changed')
        groups = sorted({(r['source'], r['class'], r['anomaly']) for r in rows})
        summarize_group = lambda subset: summarize(subset, modes)
        report = {'metadata': metadata, 'modes': list(modes),
                  'notes': ['Read-only source mechanism diagnostic; no optimization, target data or checkpoint writes.',
                            'Reuses pilot validation identities, NOT baseline-unseen validation. Image/checkpoint contents are not hashed.',
                            'Only first four visual blocks, prompt-to-CLS/patch projected Value contribution is scaled; prompt query outputs unchanged.',
                            'Q/K/softmax are unchanged at each identical layer input; earlier interventions change later layer inputs.',
                            'Contribution=0 is NOT static-only or prompt deletion. K/softmax allocation remains.',
                            'Two 0.5 controls are not equal-strength interventions. Visual amplitude control isolates branch scope.',
                            'Attention captures are after output projection; features after learned projection, BEFORE final L2 normalization.',
                            'eval/no_grad, paired RNG including HSF, aggregation=False, batch-preserving loss.',
                            'Loss and feature changes alone do not establish AP improvement or generalization.'],
                  'equivalence': {'images': len(rows), 'passed': True, 'max_abs_output_delta': maximum,
                                  'atol': 1e-6, 'rtol': 1e-5},
                  'unchanged_saved_tensors': len(before), 'unchanged_model_tensor_versions': len(versions),
                  'overall': summarize_group(rows),
                  'by_source': {s: summarize_group([r for r in rows if r['source'] == s]) for s in sorted({r['source'] for r in rows})},
                  'by_label': {str(a): summarize_group([r for r in rows if r['anomaly'] == a]) for a in sorted({r['anomaly'] for r in rows})},
                  'by_class_label': [{'source': s, 'class': c, 'anomaly': a,
                                      **summarize_group([r for r in rows if (r['source'], r['class'], r['anomaly']) == (s, c, a)])}
                                     for s, c, a in groups], 'per_image': rows}
        if layerwise:
            report['metric_protocol'] = {
                'pixel_threshold': pixel_threshold, 'positive_rule': 'score > threshold',
                'threshold_source': 'Explicit fixed input; never fitted on diagnostic images.',
                'aggregation': 'Interpolate raw two-state logits, mean layers, softmax, Gaussian sigma=4 once.',
                'ap_summary': 'Mean per-anomalous-image pixel AP in [0,1], NOT pooled/class-macro benchmark P-AP.',
                'ap_change_tolerance': AP_TOL,
                'area': 'GT foreground fraction after standard preprocessing; post-hoc grouping only.',
                'area_groups': {'small': '(0,0.01]', 'medium': '(0.01,0.05]', 'large': '(0.05,1]'}}
            report['interventions'] = {
                mode: {'family': intervention_spec(mode)[0],
                       'visual_block_1based': intervention_spec(mode)[1] + 1
                       if intervention_spec(mode)[1] is not None else None}
                for mode in modes}
            report['notes'].extend([
                'Each L1-L4 intervention modifies ONLY the indicated visual prompt injection block; text remains baseline.',
                'L1-L4 injection blocks are distinct from the visual feature readout layers recorded in metadata.',
                'Post-hoc best-layer counts use GT, retain ties and do NOT form a deployable selector or prove content-generation benefit.',
                'Only downweighting to 0.5 is probed; this does not exhaust possible layer interventions or Prompt content changes.',
                'AP is null for empty masks. Normal false positives are compared at the same supplied source threshold.',
                'Baseline per-readout metrics are descriptive; threshold was calibrated for the aggregate, not each readout.',
                'This is a repeated-forward diagnostic, not a deployment latency benchmark.'])
            report['metadata'] = {**metadata, 'visual_readout_layers': list(model.output_layers)}
            report['by_area'] = {group: summarize_group([r for r in rows if r['area_group'] == group])
                                 for group in sorted({r['area_group'] for r in rows})}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
        summary = report['overall']['detection_metrics'] if layerwise else report['overall']
        logger.info('Contribution overall: ' + json.dumps(summary, allow_nan=False))
        label = 'Prompt layer sensitivity' if layerwise else 'Prompt contribution'
        logger.info(f'{label}: DONE; equivalence=PASS; images={len(rows)}; '
                    f'unchanged_saved_tensors={len(before)}; report={path}')
        return report
    finally:
        for m, flag in flags:
            m.training = flag
        for p, flag in grad_flags:
            p.requires_grad_(flag)
        _restore_rng(rng)


def run_diagnostic(trainer, args, logger, get_data):
    logger.info('SOURCE-ONLY prompt contribution diagnostic: testing_data/source sampling flags are ignored; no target data or training.')
    loaders, metadata = source_loaders(trainer, args, get_data)
    return diagnose(trainer, loaders, logger,
                    Path(args.save_path) / 'diagnostics' / (report_name(args) + '.json'), metadata,
                    layerwise=getattr(args, 'prompt_contribution_scope', 'all') == 'layerwise',
                    pixel_threshold=getattr(args, 'contribution_pixel_threshold', None))
