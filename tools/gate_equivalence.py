"""Read-only, paired add versus shared_gate=1 inference checks."""
import random
import json
import statistics
from pathlib import Path

import numpy as np
import torch


def representative_indices(dataset, per_group=1):
    """First N images per (class, normal/anomalous), using metadata only."""
    if per_group < 1:
        raise ValueError("equivalence_per_group must be positive")
    counts, indices = {}, []
    for index, item in enumerate(dataset.data_all):
        group = (item["cls_name"], int(item["anomaly"]))
        if counts.get(group, 0) < per_group:
            indices.append(index)
            counts[group] = counts.get(group, 0) + 1
    if not indices:
        raise ValueError("No images available for equivalence checking")
    return indices


def _rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


def check_gate_equivalence(trainer, loader, logger, atol=1e-6, rtol=1e-5):
    """Use one backbone to avoid doubling GPU memory. Switch the actual add
    and shared prompt paths, bypassing the gate generator for the add pass.
    This is an inference smoke test, not a full-dataset metric comparison.
    """
    model = trainer.clip_model
    if trainer.fusion_mode != "shared_gate" or model.static_only:
        raise ValueError("Equivalence check requires shared_gate with dynamic prompts enabled")
    if atol < 0 or rtol < 0:
        raise ValueError("Equivalence tolerances must be nonnegative")
    prompters = (model.visual_prompter, model.text_prompter)
    original_modes = [model.fusion_mode] + [p.fusion_mode for p in prompters]
    gate = model.shared_gate_generator
    training_flags = [(module, module.training) for module in model.modules()]
    rng = _rng_state()
    total, failures = 0, 0
    maxima = {"map": 0.0, "score": 0.0}
    try:
        model.eval()
        with torch.no_grad():
            for items in loader:
                if items["img"].shape[0] != 1:
                    raise ValueError("Equivalence checking requires batch size 1")
                image = items["img"].to(trainer.device)
                paired_rng = _rng_state()
                model.fusion_mode = "add"
                for prompter in prompters:
                    prompter.fusion_mode = "add"
                model.shared_gate_generator = None
                reference = model(image, items["cls_name"], aggregation=True)
                _restore_rng(paired_rng)
                model.fusion_mode = "shared_gate"
                for prompter in prompters:
                    prompter.fusion_mode = "shared_gate"
                model.shared_gate_generator = gate
                candidate = model(image, items["cls_name"], aggregation=True, gate_override=1.0)
                passed, differences = True, {}
                for label, before, after in zip(("map", "score"), reference, candidate):
                    valid = (before.shape == after.shape
                             and torch.isfinite(before).all().item()
                             and torch.isfinite(after).all().item())
                    diff = (before.float() - after.float()).abs().max().item() if valid else float("inf")
                    passed = passed and valid and torch.allclose(before, after, atol=atol, rtol=rtol)
                    differences[label] = diff
                    maxima[label] = max(maxima[label], diff)
                total += 1
                failures += int(not passed)
                logger.info(
                    f'Equivalence {"PASS" if passed else "FAIL"}: '
                    f'class={items["cls_name"][0]} path={items["img_path"][0]} '
                    f'map_max_abs={differences["map"]:.8g} score_max_abs={differences["score"]:.8g}'
                )
        if not total:
            raise ValueError("No images were checked")
        logger.info(
            f'Equivalence summary: {"PASS" if failures == 0 else "FAIL"}; '
            f'images={total}, failed={failures}, map_max_abs={maxima["map"]:.8g}, '
            f'score_max_abs={maxima["score"]:.8g}, atol={atol:g}, rtol={rtol:g}'
        )
        if failures:
            raise RuntimeError("add and shared_gate=1 differ; do not proceed to gate training")
        return {"images": total, "failed": failures, **maxima}
    finally:
        model.shared_gate_generator = gate
        model.fusion_mode = original_modes[0]
        for prompter, mode in zip(prompters, original_modes[1:]):
            prompter.fusion_mode = mode
        for module, training in training_flags:
            module.training = training
        _restore_rng(rng)


def _prompt_pair_metrics(reference, candidate):
    """Tokenwise cosine and block relative L2; undefined zero norms stay null."""
    a, b = reference.float(), candidate.float()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise RuntimeError("Invalid prompt pair shape or non-finite values")
    a_norm, b_norm = a.norm(dim=-1), b.norm(dim=-1)
    denominator = a.norm().item()
    cosine = None
    if (a_norm > 1e-12).all() and (b_norm > 1e-12).all():
        cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
    return {
        "cosine_mean": cosine,
        "relative_l2": (b - a).norm().item() / denominator if denominator > 1e-12 else None,
    }


def _capture_prompt_norms(model, image, cls_names):
    """Run the actual prompted encoders; hooks are installed AFTER conditioning.

    Only freshly injected tokens are captured. In particular, text layer zero
    does not inject a prompt and is deliberately excluded.
    """
    captures, handles, expected = {}, [], set()

    def make_hook(branch, layer, prompter):
        def hook(module, inputs, output):
            length = prompter.length
            token_slice = slice(-length, None) if branch == "visual" else slice(1, 1 + length)
            before = inputs[0][token_slice, 0, :].detach().clone()
            after = output[token_slice, 0, :].detach().clone()
            composed = prompter.compose_prompt(layer, 1)[0].to(before.dtype)
            if not torch.equal(before, composed):
                raise RuntimeError(f"Captured tokens do not match injected prompts: {branch}/{layer}")
            key = (branch, layer)
            if key in captures:
                # Text encoding invokes each block for normal/abnormal templates.
                # Their freshly injected prompts must agree; count each image once.
                if not all(torch.equal(a, b) for a, b in zip(captures[key], (before, after))):
                    raise RuntimeError(f"Inconsistent repeated prompt capture: {branch}/{layer}")
            else:
                captures[key] = (before, after)
        return hook

    branches = (
        ("visual", model.visual_prompter, model.visual.transformer.resblocks),
        ("text", model.text_prompter, model.transformer.resblocks),
    )
    try:
        for branch, prompter, blocks in branches:
            if not prompter.enabled:
                continue
            if prompter.depth > len(blocks):
                raise ValueError("Prompt depth exceeds transformer depth")
            for layer in range(1 if branch == "text" else 0, prompter.depth):
                expected.add((branch, layer))
                handles.append(blocks[layer].ln_1.register_forward_hook(make_hook(branch, layer, prompter)))
        if model.visual_prompter.enabled:
            model.encode_image(image)
        if model.text_prompter.enabled:
            model.text_embedding_layer(model, cls_names, model.device)
        if set(captures) != expected or not expected:
            raise RuntimeError(f"Missing active prompt captures: {expected - set(captures)}")
        return captures
    finally:
        for handle in handles:
            handle.remove()


def diagnose_prompt_scale(trainer, loader, logger, output_path):
    """Compare gate 1 vs 0.5 at actual prompt injection/ln_1 boundaries.

    No detector losses, optimizer updates, HSF or full-dataset evaluation.
    Norm statistics describe prompts, not the complete residual/attention output.
    """
    model = trainer.clip_model
    if trainer.fusion_mode != "shared_gate" or model.prompting_type != "SD" or model.static_only:
        raise ValueError("Prompt diagnosis requires shared_gate, SD, and dynamic prompts enabled")
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"Diagnostic report already exists; use a new --save_path: {path}")
    prompters = (model.visual_prompter, model.text_prompter)
    caches = [(p, p.dynamic_prompts, p.dynamic_gates) for p in prompters]
    attributes = {name: getattr(model, name) for name in
                  ("condition_features", "shared_gate_logits", "shared_gate_weights")}
    text_cache = model.text_embedding_layer.ensemble_text_features.copy()
    flags = [(m, m.training) for m in model.modules()]
    rng = _rng_state()
    rows, images = [], 0
    try:
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=str(trainer.device).startswith("cuda")):
            for items in loader:
                if items["img"].shape[0] != 1:
                    raise ValueError("Prompt diagnosis requires batch size 1")
                image = items["img"].to(trainer.device)
                # No hooks during the frozen visual conditioning pass.
                model.generate_and_set_dynamic_prompts(image, gate_override=1.0)
                paired_rng = _rng_state()
                one = _capture_prompt_norms(model, image, items["cls_name"])
                for prompter in prompters:
                    if prompter.enabled:
                        prompter.dynamic_gates = prompter.dynamic_gates.new_full(prompter.dynamic_gates.shape, 0.5)
                _restore_rng(paired_rng)
                half = _capture_prompt_norms(model, image, items["cls_name"])
                if set(one) != set(half):
                    raise RuntimeError("Prompt capture keys changed between gates")
                for (branch, layer), (raw_one, ln_one) in one.items():
                    prompter = model.visual_prompter if branch == "visual" else model.text_prompter
                    static = prompter.static_prompts[layer].detach().float()
                    dynamic = prompter.dynamic_prompts[0].detach().float()
                    if not torch.isfinite(static).all() or not torch.isfinite(dynamic).all():
                        raise RuntimeError("Non-finite static/dynamic prompts")
                    static_norm = static.norm(dim=-1).mean().item()
                    dynamic_norm = dynamic.norm(dim=-1).mean().item()
                    raw_half, ln_half = half[(branch, layer)]
                    row = {
                        "class": items["cls_name"][0], "image": items["img_path"][0],
                        "anomaly": int(items["anomaly"][0]), "branch": branch, "layer": layer,
                        "static_norm_mean": static_norm, "dynamic_norm_mean": dynamic_norm,
                        "dynamic_static_ratio": dynamic_norm / static_norm if static_norm > 1e-12 else None,
                        "raw_dtype": str(raw_one.dtype), "ln_dtype": str(ln_one.dtype),
                    }
                    row.update({"raw_" + k: v for k, v in _prompt_pair_metrics(raw_one, raw_half).items()})
                    row.update({"ln_" + k: v for k, v in _prompt_pair_metrics(ln_one, ln_half).items()})
                    rows.append(row)
                images += 1
                logger.info(f'Prompt scale image={images} class={items["cls_name"][0]} active_layers={len(one)}')
        if not rows:
            raise ValueError("No images were diagnosed")
        keys = ("static_norm_mean", "dynamic_norm_mean", "dynamic_static_ratio",
                "raw_cosine_mean", "raw_relative_l2", "ln_cosine_mean", "ln_relative_l2")
        summaries = []
        for branch, layer in sorted({(r["branch"], r["layer"]) for r in rows}):
            group = [r for r in rows if r["branch"] == branch and r["layer"] == layer]
            summary = {"branch": branch, "layer": layer, "images": len(group)}
            for key in keys:
                values = [r[key] for r in group if r[key] is not None]
                summary[key] = ({"mean": statistics.mean(values), "median": statistics.median(values),
                                 "min": min(values), "max": max(values), "valid_count": len(values)}
                                if values else {"valid_count": 0})
            summaries.append(summary)
            logger.info('Prompt scale layer: ' + json.dumps(summary, allow_nan=False))
        report = {
            "images": images, "reference_gate": 1.0, "candidate_gate": 0.5,
            "notes": ["Layer indices are zero-based; text layer 0 is inactive and excluded.",
                      "Norm means are per-token L2 means; ratio is dynamic_norm_mean/static_norm_mean.",
                      "Cosines are tokenwise means; relative L2 is ||half-one||_F/||one||_F.",
                      "Undefined zero-norm metrics are null, never fabricated as zero.",
                      "ln metrics use actual ln_1 forward hooks, including inference dtype and learned affine parameters.",
                      "LN effects do not imply whole-model invariance: the residual bypass retains raw prompts.",
                      "Subset diagnostic only; no training, HSF, target-label optimization or task metrics."],
            "summary": summaries, "per_image": rows,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        logger.info(f'Prompt scale summary: DONE; images={images}, active_layers={len(summaries)}, report={path}')
        return report
    finally:
        for prompter, dynamic, gates in caches:
            prompter.dynamic_prompts, prompter.dynamic_gates = dynamic, gates
        for name, value in attributes.items():
            setattr(model, name, value)
        model.text_embedding_layer.ensemble_text_features = text_cache
        for module, training in flags:
            module.training = training
        _restore_rng(rng)
