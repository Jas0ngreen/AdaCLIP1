"""Read-only, paired add versus shared_gate=1 inference checks."""
import random

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
