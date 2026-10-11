"""Small source-only direct-loss experiment. No target data or model selection."""
import json
import math
import random
import statistics
from pathlib import Path

import numpy as np
import torch

from .gate_equivalence import _rng_state, _restore_rng


METRICS = ('classification', 'segmentation', 'total')
MODES = ('learned', '1', '0.5')


def split_source(dataset, source, train_count, val_count, seed):
    """Split by source/class/label; refuse duplicate paths and insufficient groups."""
    if source not in ('mvtec', 'colondb') or train_count < 1 or val_count < 1:
        raise ValueError('Invalid pilot source or sample counts')
    if getattr(dataset, 'training', True):
        raise ValueError('Pilot datasets must disable augmentation to prevent composite-image leakage')
    groups, seen = {}, set()
    for index, item in enumerate(dataset.data_all):
        path = str((Path(dataset.root) / item['img_path']).resolve())
        if path in seen:
            raise ValueError(f'Duplicate source image: {path}')
        seen.add(path)
        label = int(item['anomaly'])
        if label not in (0, 1):
            raise ValueError('Expected binary labels')
        groups.setdefault((item['cls_name'], label), []).append((index, path))
    if not groups:
        raise ValueError('Empty source dataset')
    rng = random.Random(seed)
    indices = {'train': [], 'validation': []}
    manifest = []
    for (cls, label), available in sorted(groups.items()):
        if len(available) < train_count + val_count:
            raise ValueError(f'Insufficient {source}/{cls}/{label} images for disjoint pilot subsets')
        chosen = rng.sample(available, train_count + val_count)
        row = {'source': source, 'class': cls, 'anomaly': label, 'available': len(available)}
        for name, subset in (('validation', chosen[:val_count]), ('train', chosen[val_count:])):
            row[name] = [{'index': index, 'image': path} for index, path in sorted(subset)]
            indices[name].extend(index for index, _ in subset)
        manifest.append(row)
    return {k: sorted(v) for k, v in indices.items()}, manifest


def summarize(rows):
    if not rows:
        raise ValueError('Empty pilot evaluation')
    means = {mode: {m: statistics.mean(r['losses'][mode][m] for r in rows) for m in METRICS}
             for mode in MODES}
    comparisons = {}
    for reference in ('1', '0.5'):
        comparisons[reference] = {}
        for metric in METRICS:
            pairs = [(r['losses']['learned'][metric], r['losses'][reference][metric]) for r in rows]
            tied = [math.isclose(a, b, abs_tol=1e-6, rel_tol=1e-5) for a, b in pairs]
            comparisons[reference][metric] = {
                'mean_delta_learned_minus_fixed': means['learned'][metric] - means[reference][metric],
                'improves': sum(a < b and not t for (a, b), t in zip(pairs, tied)),
                'worsens': sum(a > b and not t for (a, b), t in zip(pairs, tied)),
                'ties': sum(tied)}
    gates = [r['gate'] for r in rows]
    return {'images': len(rows), 'mean_losses': means, 'comparisons': comparisons,
            'gate': {'mean': statistics.mean(gates), 'std': statistics.pstdev(gates),
                     'min': min(gates), 'max': max(gates)}}


def evaluate_pilot(trainer, loaders, seed):
    """Paired RNG for all modes; frozen final model, no augmentation or updates."""
    rng = _rng_state()
    flags = [(m, m.training) for m in trainer.clip_model.modules()]
    before = {k: v.detach().cpu().clone() for k, v in trainer.state_dict().items()
              if any(part in k for part in trainer.learnable_paramter_list)}
    rows = []
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        trainer.clip_model.eval()
        with torch.no_grad():
            for source, loader in loaders:
                for items in loader:
                    if items['img'].shape[0] != 1:
                        raise ValueError('Pilot requires batch size one')
                    paired_rng = _rng_state()
                    losses = {}
                    for mode in MODES:
                        _restore_rng(paired_rng)
                        maps, scores = trainer.clip_model(items['img'].to(trainer.device), items['cls_name'],
                                                          aggregation=False,
                                                          gate_override=None if mode == 'learned' else float(mode))
                        if mode == 'learned':
                            gate = trainer.clip_model.shared_gate_logits.sigmoid().item()
                        targets = {**items, 'img_mask': items['img_mask'].clone(), 'anomaly': items['anomaly'].clone()}
                        values = trainer.detection_loss(maps, scores, targets, return_components=True,
                                                        preserve_batch_dim=True)
                        losses[mode] = {k: values[k].item() for k in METRICS}
                        if not all(math.isfinite(v) for v in losses[mode].values()):
                            raise RuntimeError('Non-finite pilot evaluation loss')
                        if not math.isclose(losses[mode]['total'], losses[mode]['classification'] + losses[mode]['segmentation'],
                                            abs_tol=1e-6, rel_tol=1e-5):
                            raise RuntimeError('Inconsistent pilot loss components')
                    if not math.isfinite(gate):
                        raise RuntimeError('Non-finite pilot evaluation gate')
                    rows.append({'source': source, 'class': items['cls_name'][0], 'anomaly': int(items['anomaly'][0]),
                                 'image': items['img_path'][0], 'gate': gate, 'losses': losses})
        current = trainer.state_dict()
        if any(not torch.equal(v, current[k].detach().cpu()) for k, v in before.items()):
            raise RuntimeError('Pilot evaluation changed saved parameters')
        groups = sorted({(r['source'], r['class'], r['anomaly']) for r in rows})
        return {'overall': summarize(rows),
                'by_source': {s: summarize([r for r in rows if r['source'] == s]) for s in sorted({r['source'] for r in rows})},
                'by_class_label': [{'source': s, 'class': c, 'anomaly': a,
                                    **summarize([r for r in rows if (r['source'], r['class'], r['anomaly']) == (s, c, a)])}
                                   for s, c, a in groups],
                'unchanged_saved_tensors': len(before), 'per_image': rows}
    finally:
        for module, training in flags:
            module.training = training
        _restore_rng(rng)


def run_direct_gate_pilot(trainer, args, logger, checkpoint_prefix, get_data):
    folder = Path(args.save_path) / 'diagnostics'
    report_path, split_path = folder / 'direct_gate_pilot.json', folder / 'source_split.json'
    if report_path.exists() or split_path.exists():
        raise FileExistsError('Use a new pilot output directory')
    subsets, loaders, manifest = [], {'train': [], 'validation': []}, []
    seen_paths = set()
    for source in args.training_data:
        _, dataset, _ = get_data(dataset_type_list=source, transform=trainer.preprocess,
                                  target_transform=trainer.transform, training=False)
        indices, source_manifest = split_source(dataset, source, args.pilot_train_per_group,
                                                args.pilot_val_per_group, args.pilot_split_seed)
        for row in source_manifest:
            for name in ('train', 'validation'):
                for record in row[name]:
                    if record['image'] in seen_paths:
                        raise ValueError('Overlapping pilot images across splits/sources')
                    seen_paths.add(record['image'])
        manifest.extend(source_manifest)
        for name in loaders:
            subset = torch.utils.data.Subset(dataset, indices[name])
            loaders[name].append((source, torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)))
            if name == 'train':
                subsets.append(subset)
        logger.info(f'Pilot source={source}: train={len(indices["train"])} validation={len(indices["validation"])}; augmentation=off')
    metadata = {'baseline_checkpoint': args.gate_from_baseline, 'seed': args.seed,
                'split_seed': args.pilot_split_seed, 'train_per_group': args.pilot_train_per_group,
                'validation_per_group': args.pilot_val_per_group, 'gate_learning_rate': args.gate_learning_rate,
                'epochs': 1, 'image_size': args.image_size, 'use_hsf': args.use_hsf,
                'k_clusters': args.k_clusters, 'model': args.model, 'sources': args.training_data,
                'prompting_type': args.prompting_type, 'prompting_branch': args.prompting_branch,
                'prompting_depth': args.prompting_depth, 'prompting_length': args.prompting_length,
                'objective': 'direct_classification_plus_segmentation', 'augmentation': False,
                'final_checkpoint': checkpoint_prefix + '_final.pth'}
    folder.mkdir(parents=True, exist_ok=True)
    with split_path.open('x', encoding='utf8') as f:
        json.dump({'metadata': metadata, 'groups': manifest}, f, indent=2, ensure_ascii=False)
    trainer.save(checkpoint_prefix + '_base.pth')
    trainer.prepare_shared_gate_training()
    logger.info('Direct gate pilot: pure classification + batch-preserving segmentation loss; no BCE or endpoint teacher.')
    train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(subsets), batch_size=1, shuffle=True)
    loss = trainer.train_epoch(train_loader, stage='gate_direct')
    frozen = trainer.assert_shared_gate_frozen()
    logger.info(f'Freeze check PASS: {frozen} non-gate checkpoint tensors unchanged.')
    logger.info(f'Direct gate epoch [1/1], loss={loss:.8f}; ' + json.dumps(trainer.last_gate_training_statistics, allow_nan=False))
    trainer.save(checkpoint_prefix + '_final.pth')
    report = {'metadata': metadata, 'split_manifest': str(split_path),
              'online_training': trainer.last_gate_training_statistics,
              'notes': ['No target dataset loaded or used for checkpoint selection.',
                        'Source images are held out from this gate fit only; the original baseline may have trained on them.',
                        'Train and validation both disable augmentation to prevent composite-image split leakage.',
                        'Post-fit evaluation uses a fixed final gate on both subsets, paired RNG, aggregation=False and batch-preserving loss.',
                        'Sampled-image means are not population-weighted estimates, AUROC/AP or evidence of target generalization.',
                        'Exactly one epoch; no automatic extension, model selection or validation-based fixed-gate deployment choice.']}
    for name in ('train', 'validation'):
        logger.info(f'Pilot post-fit evaluation: {name}; learned vs fixed 1 and 0.5')
        report[name] = evaluate_pilot(trainer, loaders[name], args.seed)
        logger.info(f'Pilot {name} summary: ' + json.dumps(report[name]['overall'], allow_nan=False))
    trainer.assert_shared_gate_frozen()
    with report_path.open('x', encoding='utf8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
    logger.info(f'Direct gate pilot: DONE; report={report_path}; no target evaluation performed.')
    return report
