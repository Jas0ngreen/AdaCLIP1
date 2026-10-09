"""Source-only, paired loss diagnosis. No optimization or checkpoint writes."""
import json
import math
import random
import statistics
from itertools import combinations
from pathlib import Path

import torch

from .gate_equivalence import _rng_state, _restore_rng


GATES = (1.0, 0.5, 0.01)
METRICS = ("classification", "segmentation", "total")
ABS_TOL, REL_TOL = 1e-6, 1e-5


def select_source_samples(dataset, per_group=2, seed=111):
    """Seeded sampling per (class, label), without decoding any images."""
    if per_group < 1:
        raise ValueError("source_samples_per_group must be positive")
    groups = {}
    for index, item in enumerate(dataset.data_all):
        label = int(item["anomaly"])
        if label not in (0, 1):
            raise ValueError("Expected source labels 0/1")
        groups.setdefault((item["cls_name"], label), []).append(index)
    if not groups:
        raise ValueError("Empty source dataset")
    rng = random.Random(seed)
    indices, coverage = [], []
    for (name, label), available in sorted(groups.items()):
        selected = rng.sample(available, min(per_group, len(available)))
        indices.extend(selected)
        coverage.append({"class": name, "anomaly": label, "available": len(available),
                         "selected_indices": sorted(selected), "selected": len(selected)})
    return sorted(indices), coverage


def _preference(candidate, reference):
    if math.isclose(candidate, reference, rel_tol=REL_TOL, abs_tol=ABS_TOL):
        return "tie"
    return "improves" if candidate < reference else "worsens"


def annotate_losses(losses):
    """Keep all tied minima; report both endpoint comparisons and all pairs."""
    preferred = {}
    for metric in METRICS:
        minimum = min(values[metric] for values in losses.values())
        preferred[metric] = [gate for gate, values in losses.items()
                             if _preference(values[metric], minimum) == "tie"]
    paired = []
    for reference, candidate in combinations(losses, 2):
        directions = {metric: _preference(losses[candidate][metric], losses[reference][metric])
                      for metric in METRICS}
        paired.append({
            "reference_gate": reference, "candidate_gate": candidate,
            "delta": {metric: losses[candidate][metric] - losses[reference][metric] for metric in METRICS},
            "direction": directions,
            "task_conflict": {directions["classification"], directions["segmentation"]} == {"improves", "worsens"},
            "total_improves_but_classification_worsens": directions["total"] == "improves" and directions["classification"] == "worsens",
            "total_improves_but_segmentation_worsens": directions["total"] == "improves" and directions["segmentation"] == "worsens",
        })
    return {"preferred_gates": preferred,
            "disjoint_task_minima": not bool(set(preferred["classification"]) & set(preferred["segmentation"])),
            "comparisons": paired}


def summarize_losses(rows):
    if not rows:
        raise ValueError("No source images were diagnosed")
    gates = [f"{gate:g}" for gate in GATES]
    result = {"images": len(rows),
              "disjoint_task_minima": sum(r["disjoint_task_minima"] for r in rows),
              "mean_losses": {g: {m: statistics.mean(r["losses"][g][m] for r in rows)
                                  for m in METRICS} for g in gates},
              "tie_inclusive_preferred_counts": {
                  m: {g: sum(g in r["preferred_gates"][m] for r in rows) for g in gates} for m in METRICS}}
    result["comparisons"] = []
    flags = ("task_conflict", "total_improves_but_classification_worsens", "total_improves_but_segmentation_worsens")
    for index, (reference, candidate) in enumerate(combinations(gates, 2)):
        pairs = [r["comparisons"][index] for r in rows]
        result["comparisons"].append({
            "reference_gate": reference, "candidate_gate": candidate,
            "mean_delta": {m: statistics.mean(p["delta"][m] for p in pairs) for m in METRICS},
            "counts": {m: {d: sum(p["direction"][m] == d for p in pairs)
                           for d in ("improves", "tie", "worsens")} for m in METRICS},
            **{flag: sum(p[flag] for p in pairs) for flag in flags},
        })
    return result


def diagnose_source_losses(trainer, source_loaders, logger, output_path, metadata):
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"Report already exists; use a new --save_path: {path}")
    if trainer.fusion_mode != "shared_gate":
        raise ValueError("Source loss diagnosis requires shared_gate")
    source_names = [name for name, _ in source_loaders]
    if not source_names or len(source_names) != len(set(source_names)) or not set(source_names) <= {"mvtec", "colondb"}:
        raise ValueError("This source diagnosis supports distinct mvtec/colondb sources only")
    model = trainer.clip_model
    flags = [(module, module.training) for module in model.modules()]
    rng = _rng_state()
    before = {name: value.detach().cpu().clone() for name, value in trainer.state_dict().items()
              if any(part in name for part in trainer.learnable_paramter_list)}
    rows = []
    try:
        model.eval()
        with torch.no_grad():
            for source, loader in source_loaders:
                source_count = 0
                for items in loader:
                    if items["img"].shape[0] != 1:
                        raise ValueError("Source diagnosis requires batch size 1")
                    image = items["img"].to(trainer.device)
                    paired_rng = _rng_state()
                    losses = {}
                    for gate in GATES:
                        _restore_rng(paired_rng)
                        maps, scores = model(image, items["cls_name"], aggregation=False, gate_override=gate)
                        # The legacy loss thresholds targets in-place; isolate each pass.
                        targets = {**items, "img_mask": items["img_mask"].clone(), "anomaly": items["anomaly"].clone()}
                        components = trainer.detection_loss(maps, scores, targets, return_components=True)
                        values = {name: components[name].item() for name in METRICS}
                        if not all(math.isfinite(v) for v in values.values()):
                            raise RuntimeError(f"Non-finite losses: {source}, {items['img_path'][0]}, gate={gate}")
                        if not math.isclose(values["total"], values["classification"] + values["segmentation"], rel_tol=REL_TOL, abs_tol=ABS_TOL):
                            raise RuntimeError("Loss components do not sum to total")
                        losses[f"{gate:g}"] = values
                    row = {"source": source, "class": items["cls_name"][0], "anomaly": int(items["anomaly"][0]),
                           "image": items["img_path"][0], "losses": losses, **annotate_losses(losses)}
                    rows.append(row)
                    source_count += 1
                    logger.info(f'Source loss image={len(rows)} source={source} class={row["class"]} anomaly={row["anomaly"]} '
                                f'preferred={row["preferred_gates"]} disjoint_task_minima={row["disjoint_task_minima"]}')
                if source_count == 0:
                    raise ValueError(f"No images diagnosed for source: {source}")
        current = trainer.state_dict()
        changed = [name for name, value in before.items() if not torch.equal(value, current[name].detach().cpu())]
        if changed:
            raise RuntimeError(f"Saved detector/gate parameters changed during diagnosis: {changed}")
        groups = sorted({(r["source"], r["class"], r["anomaly"]) for r in rows})
        report = {"gates": list(GATES), "metadata": metadata,
                  "comparison_tolerance": {"abs": ABS_TOL, "rel": REL_TOL},
                  "notes": ["Source data only. No VisA, optimization, gate selection for deployment, or checkpoint writes.",
                            "Existing source meta['test'] population is used with training=False to disable augmentation, not a held-out validation set.",
                            "Exact legacy detection_loss is decomposed: classification Focal + sum of per-layer pixel Focal and two Dice terms.",
                            "Legacy gt.squeeze() / Dice reduction semantics are deliberately preserved; no loss correction is made here.",
                            "eval mode, no_grad and aggregation=False match gate utility targets; HSF is retained if enabled, with paired RNG.",
                            "Preferred gates are per-image loss minima with ties, NOT AUROC/AP optima or deployable oracle performance.",
                            "Summary means are sampled-image means, not population-weighted dataset estimates; no causal generalization claim."],
                  "overall": summarize_losses(rows),
                  "by_source": {source: summarize_losses([r for r in rows if r["source"] == source]) for source in source_names},
                  "by_class_label": [{"source": s, "class": c, "anomaly": a,
                                      **summarize_losses([r for r in rows if (r["source"], r["class"], r["anomaly"]) == (s, c, a)])}
                                     for s, c, a in groups],
                  "unchanged_saved_tensors": len(before), "per_image": rows}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        for source, summary in report["by_source"].items():
            logger.info(f'Source loss summary {source}: ' + json.dumps(summary, allow_nan=False))
        logger.info(f'Source loss summary: DONE; images={len(rows)}, unchanged_saved_tensors={len(before)}, report={path}')
        return report
    finally:
        for module, training in flags:
            module.training = training
        _restore_rng(rng)
