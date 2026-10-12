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
METRICS = ('classification', 'segmentation', 'total')


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
    if (Path(args.save_path) / 'diagnostics/source_prompt_contribution.json').exists():
        raise FileExistsError('Use a new --save_path; an existing report will not be overwritten')


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
def capture_intervention(model, mode):
    """Do not intercept the initial, unprompted image-conditioning forward."""
    if mode not in MODES:
        raise ValueError(mode)
    p = model.visual_prompter
    blocks = model.visual.transformer.resblocks
    captures, handles = {}, []
    active = [None]
    old_gates = [None]
    prompters = (p, model.text_prompter)
    original_modes = [model.fusion_mode] + [x.fusion_mode for x in prompters]
    generator = model.shared_gate_generator

    def before_prompt(module, inputs):
        block, index = inputs[:2]
        if active[0] is not None:
            raise RuntimeError('Unexpected nested visual prompter')
        active[0] = index
        if mode == 'amplitude_visual_0.5':
            old_gates[0] = module.dynamic_gates
            module.dynamic_gates = torch.full_like(module.dynamic_gates, 0.5)

    def after_prompt(module, inputs, output):
        if mode == 'amplitude_visual_0.5':
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
            if mode.startswith('contribution_'):
                contribution = prompt_value_contribution(module, inputs, p.length)
                captures['message_visual_' + str(index)] = contribution.detach().float().cpu()
                if mode == 'contribution_0.5':
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
        if not required <= set(captures) or active[0] is not None:
            raise RuntimeError('Incomplete prompted-path captures')
    finally:
        for handle in handles:
            handle.remove()
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


def summarize(rows):
    means = {mode: {metric: statistics.mean(r['losses'][mode][metric] for r in rows)
                    for metric in METRICS} for mode in MODES}
    comparisons = {}
    for mode in MODES[2:]:
        comparisons[mode] = {}
        for metric in METRICS:
            deltas = [r['losses'][mode][metric] - r['losses']['baseline'][metric] for r in rows]
            ties = [math.isclose(r['losses'][mode][metric], r['losses']['baseline'][metric],
                                 abs_tol=1e-6, rel_tol=1e-5) for r in rows]
            comparisons[mode][metric] = {'mean_delta': statistics.mean(deltas),
                'improves': sum(d < 0 and not t for d, t in zip(deltas, ties)),
                'worsens': sum(d > 0 and not t for d, t in zip(deltas, ties)), 'ties': sum(ties)}
    changes = {}
    for mode in MODES[2:]:
        changes[mode] = {}
        for key in rows[0]['changes'][mode]:
            values = [r['changes'][mode][key]['relative_l2'] for r in rows]
            valid = [v for v in values if v is not None]
            changes[mode][key] = {'mean_relative_l2': statistics.mean(valid) if valid else None,
                                  'valid_count': len(valid)}
    return {'images': len(rows), 'mean_losses': means, 'vs_baseline': comparisons,
            'representation_changes': changes}


def diagnose(trainer, loaders, logger, output_path, metadata):
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(path)
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
                    losses, changes, messages = {}, {}, {}
                    reference_capture, reference_output = None, None
                    for mode in MODES:
                        _restore_rng(paired_rng)
                        gate = None if mode == 'baseline' else (0.5 if mode == 'amplitude_shared_0.5' else 1.0)
                        with capture_intervention(model, mode) as captures:
                            maps, scores = model(items['img'].to(trainer.device), items['cls_name'],
                                                 aggregation=False, gate_override=gate)
                        outputs = [m.detach().float().cpu() for m in maps] + [scores.detach().float().cpu()]
                        if any(not torch.isfinite(x).all() for x in outputs):
                            raise RuntimeError('Non-finite model output')
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
                                message = captures['message_visual_' + str(i)]
                                out = captures['attention_visual_' + str(i)]
                                # Reconstruct the local, ungated output at this SAME layer input.
                                original = out + (0.5 * message if mode == 'contribution_0.5' else 0)
                                denominator = original.norm().item()
                                messages[mode][str(i)] = {'norm': message.norm().item(),
                                    'relative_to_local_attention': message.norm().item() / denominator
                                    if denominator > 1e-12 else None}
                    row = {'source': source, 'class': items['cls_name'][0],
                           'anomaly': int(items['anomaly'][0]), 'image': items['img_path'][0],
                           'losses': losses, 'changes': changes, 'prompt_messages': messages}
                    rows.append(row)
                    logger.info(f'Contribution image={len(rows)} source={source} class={row["class"]} '
                                f'anomaly={row["anomaly"]} equivalence=PASS totals=' +
                                json.dumps({m: losses[m]['total'] for m in MODES}))
        if not rows or len(rows) != metadata['images']:
            raise RuntimeError('Reference image count mismatch')
        current = trainer.state_dict()
        if any(not torch.equal(v, current[k].detach().cpu()) for k, v in before.items()):
            raise RuntimeError('Saved detector/gate tensors changed')
        if any(v._version != versions[k] for k, v in model.state_dict().items()):
            raise RuntimeError('Model tensor version changed')
        groups = sorted({(r['source'], r['class'], r['anomaly']) for r in rows})
        report = {'metadata': metadata, 'modes': list(MODES),
                  'notes': ['Read-only source mechanism diagnostic; no optimization, target data or checkpoint writes.',
                            'Reuses pilot validation identities, NOT baseline-unseen validation. Image/checkpoint contents are not hashed.',
                            'Only first four visual blocks, prompt-to-CLS/patch projected Value contribution is scaled; prompt query outputs unchanged.',
                            'Q/K/softmax are unchanged at each identical layer input; earlier interventions change later layer inputs.',
                            'Contribution=0 is NOT static-only or prompt deletion. K/softmax allocation remains.',
                            'Two 0.5 controls are not equal-strength interventions. Visual amplitude control isolates branch scope.',
                            'Attention captures are after output projection; features after learned projection, BEFORE final L2 normalization.',
                            'eval/no_grad, paired RNG including HSF, aggregation=False, batch-preserving loss.',
                            'Loss and feature changes are not AUROC/AP gains or proof of generalization.'],
                  'equivalence': {'images': len(rows), 'passed': True, 'max_abs_output_delta': maximum,
                                  'atol': 1e-6, 'rtol': 1e-5},
                  'unchanged_saved_tensors': len(before), 'unchanged_model_tensor_versions': len(versions),
                  'overall': summarize(rows),
                  'by_source': {s: summarize([r for r in rows if r['source'] == s]) for s in sorted({r['source'] for r in rows})},
                  'by_label': {str(a): summarize([r for r in rows if r['anomaly'] == a]) for a in sorted({r['anomaly'] for r in rows})},
                  'by_class_label': [{'source': s, 'class': c, 'anomaly': a,
                                      **summarize([r for r in rows if (r['source'], r['class'], r['anomaly']) == (s, c, a)])}
                                     for s, c, a in groups], 'per_image': rows}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
        logger.info('Contribution overall: ' + json.dumps(report['overall'], allow_nan=False))
        logger.info(f'Prompt contribution: DONE; equivalence=PASS; images={len(rows)}; '
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
                    Path(args.save_path) / 'diagnostics/source_prompt_contribution.json', metadata)
