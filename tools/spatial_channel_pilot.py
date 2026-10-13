"""Source-only feature adaptation pilot. One epoch, two trained controls."""
import json
import math
from pathlib import Path
import statistics

import torch

from method.spatial_channel import configure_adapter
from .gate_equivalence import _rng_state, _restore_rng


MODES = ('baseline', 'attention', 'gated')
METRICS = ('classification', 'segmentation', 'total')
CONFIG_KEYS = ('model', 'image_size', 'use_hsf', 'k_clusters', 'prompting_type',
               'prompting_branch', 'prompting_depth', 'prompting_length')


def validate(args):
    if (args.fusion_mode != 'add' or args.batch_size != 1 or args.epoch != 1
            or args.prompting_type != 'SD' or args.prompting_branch != 'VL'
            or args.adapter_layer != 24 or args.adapter_reduction < 1
            or not math.isfinite(args.adapter_learning_rate) or args.adapter_learning_rate <= 0
            or getattr(args, 'direct_gate_pilot', False) or args.gate_from_baseline
            or args.resume_gate_from or args.ckt_path):
        raise ValueError('Feature pilot requires add/SD/VL, layer 24, batch 1, epoch 1, positive adapter LR/reduction '
                         'and no other checkpoint/gate mode')
    baseline = Path(args.adapter_baseline).expanduser().resolve()
    reference = Path(args.adapter_reference_report).expanduser().resolve()
    if not args.adapter_baseline or not baseline.is_file():
        raise FileNotFoundError('Supply an existing --adapter_baseline')
    if not args.adapter_reference_report or not reference.is_file():
        raise FileNotFoundError('Supply the original --adapter_reference_report direct_gate_pilot.json')
    destination = Path(args.save_path).expanduser().resolve()
    if destination in baseline.parents or destination in reference.parents:
        raise ValueError('Use a new output directory outside the input experiment')
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError('Use a new or empty --save_path; previous experiments are preserved')
    with reference.open(encoding='utf8') as f:
        report = json.load(f)
    meta = report['metadata']
    if Path(meta['baseline_checkpoint']).resolve() != baseline:
        raise ValueError('Baseline path differs from the original pilot')
    for key in CONFIG_KEYS:
        if meta.get(key) != getattr(args, key):
            raise ValueError('Pilot configuration mismatch: ' + key)
    sources = meta['sources']
    if (not sources or len(sources) != len(set(sources)) or not set(sources) <= {'mvtec', 'colondb'}
            or list(args.training_data) != sources):
        raise ValueError('Training sources must match the original mvtec/colondb pilot')
    identities = set()
    for split in ('train', 'validation'):
        rows = report[split]['per_image']
        if not rows or len(rows) != report[split]['overall']['images'] or {r['source'] for r in rows} != set(sources):
            raise ValueError('Invalid reference subset: ' + split)
        for row in rows:
            path = str(Path(row['image']).resolve())
            if path in identities or row['anomaly'] not in (0, 1):
                raise ValueError('Duplicate/overlapping source image or invalid label')
            identities.add(path)
    args.adapter_baseline, args.adapter_reference_report = str(baseline), str(reference)
    return report


def make_loaders(trainer, args, reference, get_data):
    loaders, manifest = {'train': [], 'validation': []}, {'train': [], 'validation': []}
    for source in reference['metadata']['sources']:
        _, dataset, _ = get_data(dataset_type_list=source, transform=trainer.preprocess,
                                target_transform=trainer.transform, training=False)
        if getattr(dataset, 'training', True):
            raise ValueError('Source augmentation must be disabled')
        mapping = {}
        for index, item in enumerate(dataset.data_all):
            path = str((Path(dataset.root) / item['img_path']).resolve())
            if path in mapping:
                raise ValueError('Duplicate source metadata path: ' + path)
            mapping[path] = (index, item)
        for split in loaders:
            indices = []
            for row in (r for r in reference[split]['per_image'] if r['source'] == source):
                path = str(Path(row['image']).resolve())
                if path not in mapping:
                    raise FileNotFoundError('Recorded source image missing from metadata: ' + path)
                index, item = mapping[path]
                if item['cls_name'] != row['class'] or int(item['anomaly']) != row['anomaly']:
                    raise ValueError('Recorded source class/label mismatch: ' + path)
                indices.append(index)
                manifest[split].append({'source': source, 'class': row['class'], 'anomaly': row['anomaly'],
                                        'image': path, 'index': index})
            loaders[split].append((source, torch.utils.data.DataLoader(
                torch.utils.data.Subset(dataset, indices), batch_size=1, shuffle=False)))
    return loaders, manifest


def import_baseline(trainer, path):
    """Require every saved original detector tensor, with no adapter/gate weights."""
    state = torch.load(path, map_location='cpu')
    current = trainer.state_dict()
    expected = {k for k in current if any(part in k for part in trainer.base_parameter_names)}
    if set(state) != expected:
        raise ValueError('Expected a complete original add checkpoint without gate/adapter weights')
    if any(not isinstance(v, torch.Tensor) or v.shape != current[k].shape or v.dtype != current[k].dtype
           or not torch.isfinite(v).all() for k, v in state.items()):
        raise ValueError('Invalid baseline tensor')
    trainer.load_state_dict(state, strict=False)
    if any(not torch.equal(v, trainer.state_dict()[k].detach().cpu()) for k, v in state.items()):
        raise RuntimeError('Baseline import is not exact')
    return {k: v.clone() for k, v in state.items()}


def frozen_guard(trainer, before, versions, optimizer=None):
    adapter = getattr(trainer.clip_model, 'spatial_channel_adapter', None)
    allowed = {id(p) for p in adapter.parameters()} if optimizer is not None else set()
    if optimizer is not None:
        ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        if ids != allowed:
            raise RuntimeError('Optimizer must contain exactly the feature adapter parameters')
    if any(id(p) not in allowed and (p.requires_grad or p.grad is not None)
           for p in trainer.clip_model.parameters()):
        raise RuntimeError('Original detector is not fully frozen')
    current = trainer.state_dict()
    if any(not torch.equal(v, current[k].detach().cpu()) for k, v in before.items()):
        raise RuntimeError('Original saved detector tensors changed')
    if any(trainer.clip_model.state_dict()[k]._version != version for k, version in versions.items()):
        raise RuntimeError('Frozen backbone/detector tensor version changed')
    return len(before)


def losses(trainer, items):
    # Caller sets grad context. AMP is provided by the existing model forward.
    maps, scores = trainer.clip_model(items['img'].to(trainer.device), items['cls_name'], aggregation=False)
    targets = {**items, 'img_mask': items['img_mask'].clone(), 'anomaly': items['anomaly'].clone()}
    values = trainer.detection_loss(maps, scores, targets, return_components=True, preserve_batch_dim=True)
    if not all(torch.isfinite(values[k]).all() for k in METRICS):
        raise RuntimeError('Non-finite feature pilot loss')
    if not math.isclose(values['total'].item(), values['classification'].item() + values['segmentation'].item(),
                        rel_tol=1e-5, abs_tol=1e-6):
        raise RuntimeError('Loss components do not sum to total')
    return values, list(maps) + [scores]


def check_initial_equivalence(trainer, items, adapters):
    rng = _rng_state()
    maximum = 0.
    try:
        with torch.no_grad():
            reference = None
            for mode in MODES:
                _restore_rng(rng)
                trainer.clip_model.spatial_channel_adapter = adapters.get(mode)
                _, outputs = losses(trainer, items)
                outputs = [v.detach().float().cpu() for v in outputs]
                if mode == 'baseline':
                    reference = outputs
                else:
                    if len(outputs) != len(reference):
                        raise RuntimeError('Initial output count mismatch')
                    for a, b in zip(reference, outputs):
                        if a.shape != b.shape or not torch.allclose(a, b, atol=1e-6, rtol=1e-5):
                            raise RuntimeError('Zero-eta adapter baseline equivalence FAILED')
                        maximum = max(maximum, (a - b).abs().max().item())
        return {'passed': True, 'max_abs_output_delta': maximum, 'images': 1,
                'atol': 1e-6, 'rtol': 1e-5}
    finally:
        trainer.clip_model.spatial_channel_adapter = None
        _restore_rng(rng)


def train_adapter(trainer, adapter, loader, args, logger, setup_seed, before, versions):
    trainer.clip_model.spatial_channel_adapter = adapter
    trainer.clip_model.eval()
    for p in adapter.parameters():
        p.requires_grad_(True)
        p.grad = None
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.adapter_learning_rate,
                                  betas=(0.5, 0.999), weight_decay=0.)
    frozen_guard(trainer, before, versions, optimizer)
    initial = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
    setup_seed(args.seed)  # Identical shuffle/RNG sequence for the two controls.
    records = []
    for step, items in enumerate(loader, 1):
        with torch.enable_grad():
            values, _ = losses(trainer, items)
            optimizer.zero_grad(set_to_none=True)
            values['total'].backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in adapter.parameters()):
                raise RuntimeError('Missing/non-finite adapter gradient')
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in adapter.parameters()):
                raise RuntimeError('Non-finite adapter parameter after optimizer step')
        records.append({**{m: values[m].item() for m in METRICS}, **adapter.last_statistics})
        if step == 1 or step % 16 == 0 or step == len(loader):
            logger.info(f'Feature pilot {adapter.mode} step={step}/{len(loader)} total={values["total"].item():.8f} '
                        f'eta={adapter.eta.item():.8f}')
    if not records:
        raise ValueError('Empty adapter training loader')
    frozen = frozen_guard(trainer, before, versions, optimizer)
    final = adapter.state_dict()
    changed = [k for k, v in initial.items() if not torch.equal(v, final[k].detach().cpu())]
    if not changed:
        raise RuntimeError('Feature adapter did not update')
    for p in adapter.parameters():
        p.requires_grad_(False)
        p.grad = None
    return {'images': len(records), 'online_mean': {k: statistics.mean(r[k] for r in records) for k in records[0]},
            'final_eta': adapter.eta.item(), 'changed_adapter_tensors': changed,
            'unchanged_detector_tensors': frozen, 'parameters': sum(p.numel() for p in adapter.parameters())}


def summarize(rows):
    means = {mode: {m: statistics.mean(r['losses'][mode][m] for r in rows) for m in METRICS} for mode in MODES}
    comparisons = {}
    for a, b in (('attention', 'baseline'), ('gated', 'baseline'), ('gated', 'attention')):
        comparisons[a + '_minus_' + b] = {}
        for m in METRICS:
            pairs = [(r['losses'][a][m], r['losses'][b][m]) for r in rows]
            ties = [math.isclose(x, y, abs_tol=1e-6, rel_tol=1e-5) for x, y in pairs]
            comparisons[a + '_minus_' + b][m] = {
                'mean_delta': means[a][m] - means[b][m],
                'improves': sum(x < y and not t for (x, y), t in zip(pairs, ties)),
                'worsens': sum(x > y and not t for (x, y), t in zip(pairs, ties)), 'ties': sum(ties)}
    stats = {mode: {k: statistics.mean(r['adapter_statistics'][mode][k] for r in rows)
                   for k in rows[0]['adapter_statistics'][mode]} for mode in MODES[1:]}
    return {'images': len(rows), 'mean_losses': means, 'comparisons': comparisons,
            'mean_per_image_adapter_statistics': stats}


def evaluate(trainer, loaders, adapters, seed, setup_seed, logger):
    rng = _rng_state()
    flags = [(m, m.training) for m in trainer.clip_model.modules()]
    original = getattr(trainer.clip_model, 'spatial_channel_adapter', None)
    rows = []
    snapshots = {m: {k: v.detach().cpu().clone() for k, v in a.state_dict().items()} for m, a in adapters.items()}
    try:
        setup_seed(seed)
        with torch.no_grad():
            for source, loader in loaders:
                for items in loader:
                    paired = _rng_state()
                    record = {'source': source, 'class': items['cls_name'][0], 'anomaly': int(items['anomaly'][0]),
                              'image': items['img_path'][0], 'losses': {}, 'adapter_statistics': {}}
                    for mode in MODES:
                        _restore_rng(paired)
                        trainer.clip_model.spatial_channel_adapter = adapters.get(mode)
                        trainer.clip_model.eval()
                        values, _ = losses(trainer, items)
                        record['losses'][mode] = {m: values[m].item() for m in METRICS}
                        if mode != 'baseline':
                            record['adapter_statistics'][mode] = dict(adapters[mode].last_statistics)
                    rows.append(record)
                    logger.info(f'Feature validation image={len(rows)} source={source} class={record["class"]} '
                                f'anomaly={record["anomaly"]} totals=' + json.dumps({m: record['losses'][m]['total'] for m in MODES}))
        if not rows:
            raise ValueError('Empty feature validation set')
        for mode, a in adapters.items():
            if any(not torch.equal(v, a.state_dict()[k].detach().cpu()) for k, v in snapshots[mode].items()):
                raise RuntimeError('Validation changed adapter parameters')
        groups = sorted({(r['source'], r['class'], r['anomaly']) for r in rows})
        return {'overall': summarize(rows),
                'by_source': {s: summarize([r for r in rows if r['source'] == s]) for s in sorted({r['source'] for r in rows})},
                'by_label': {str(a): summarize([r for r in rows if r['anomaly'] == a]) for a in sorted({r['anomaly'] for r in rows})},
                'by_class_label': [{'source': s, 'class': c, 'anomaly': a,
                    **summarize([r for r in rows if (r['source'], r['class'], r['anomaly']) == (s, c, a)])} for s, c, a in groups],
                'per_image': rows}
    finally:
        trainer.clip_model.spatial_channel_adapter = original
        for m, flag in flags:
            m.training = flag
        _restore_rng(rng)


def run_pilot(args, get_data, trainer_class, logger_class, setup_seed):
    reference = validate(args)  # Before writing any output or loading target data.
    setup_seed(args.seed)
    folder = Path(args.save_path)
    (folder / 'logs').mkdir(parents=True, exist_ok=True)
    logger = logger_class(str(folder / 'logs/spatial_channel_pilot.txt'))
    for key, value in sorted(vars(args).items()):
        logger.info(f'{key} = {value}')
    logger.info('SOURCE-ONLY: exact previous pilot subsets, augmentation off; no target evaluation or model selection.')
    with open(Path('model_configs') / (args.model + '.json')) as f:
        config = json.load(f)
    substage = config['vision_cfg']['layers'] // 4
    trainer = trainer_class(backbone=args.model, feat_list=[substage * i for i in (1, 2, 3, 4)],
        input_dim=config['vision_cfg']['width'], output_dim=config['embed_dim'], learning_rate=0.,
        device='cuda' if torch.cuda.is_available() else 'cpu', image_size=args.image_size,
        prompting_depth=args.prompting_depth, prompting_length=args.prompting_length,
        prompting_branch=args.prompting_branch, prompting_type=args.prompting_type,
        use_hsf=args.use_hsf, k_clusters=args.k_clusters, fusion_mode='add')
    before = import_baseline(trainer, args.adapter_baseline)
    logger.info(f'Baseline import PASS: {len(before)} detector tensors copied exactly.')
    for p in trainer.clip_model.parameters():
        p.requires_grad_(False)
        p.grad = None
    trainer.clip_model.eval()
    versions = {k: v._version for k, v in trainer.clip_model.state_dict().items()}
    loaders, manifest = make_loaders(trainer, args, reference, get_data)
    for split, records in manifest.items():
        logger.info(f'Source subset {split}: images={len(records)}; exact paths/classes/labels verified.')
    adapters = {}
    for mode in MODES[1:]:
        setup_seed(args.seed)
        adapters[mode] = configure_adapter(trainer, mode, args.adapter_layer, args.adapter_reduction)
    common = adapters['attention'].state_dict()
    adapters['gated'].load_state_dict(common, strict=False)
    equivalence = check_initial_equivalence(trainer, next(iter(loaders['validation'][0][1])), adapters)
    logger.info('Initial equivalence PASS: ' + json.dumps(equivalence))
    (folder / 'diagnostics').mkdir(exist_ok=True)
    with (folder / 'diagnostics/source_split.json').open('x', encoding='utf8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    (folder / 'models').mkdir(exist_ok=True)
    report = {'metadata': {k: getattr(args, k) for k in CONFIG_KEYS},
              'notes': ['Spatial/channel residual adaptation inspired by HEPE++ Eq. (3)-(8); existing CLIP features only, no DINO/AFS/text agent.',
                        'One epoch per control; same original detector, common adapter initialization, shuffle seed and batch-preserving loss.',
                        'Adapter acts on projected layer-24 patches before final L2 normalization; CLS and other output layers unchanged directly.',
                        'Eta starts at zero. The first step can update eta; gate/attention gradients can initially be zero.',
                        'The gated control has additional parameters; this is not a parameter-matched architecture comparison.',
                        'Exact previous source subsets; held out from adapter fitting only, not baseline-unseen validation.',
                        'No target loading, checkpoint selection or automatic extension; sampled losses are not AUROC/AP or generalization evidence.',
                        'Image/checkpoint file contents are not hashed. Keep original baseline and source data unchanged.'],
              'initial_equivalence': equivalence, 'training': {}}
    report['metadata'].update(baseline_checkpoint=args.adapter_baseline,
        reference_report=args.adapter_reference_report, seed=args.seed, epochs_per_control=1,
        adapter_layer=args.adapter_layer, adapter_reduction=args.adapter_reduction,
        adapter_learning_rate=args.adapter_learning_rate, train_images=len(manifest['train']),
        validation_images=len(manifest['validation']))
    datasets = [loader.dataset for _, loader in loaders['train']]
    train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(datasets), batch_size=1, shuffle=True)
    for mode, adapter in adapters.items():
        stats = train_adapter(trainer, adapter, train_loader, args, logger, setup_seed, before, versions)
        checkpoint = folder / 'models' / (mode + '_final.pth')
        trainer.save(str(checkpoint))
        sidecar = {**report['metadata'], 'feature_adapter': mode, 'parameters': stats['parameters']}
        with checkpoint.with_suffix('.json').open('x', encoding='utf8') as f:
            json.dump(sidecar, f, indent=2, ensure_ascii=False, allow_nan=False)
        report['training'][mode] = {**stats, 'checkpoint': str(checkpoint)}
        logger.info(f'Freeze check PASS: {stats["unchanged_detector_tensors"]} detector tensors unchanged; mode={mode}; '
                    + json.dumps(stats, allow_nan=False))
    report['validation'] = evaluate(trainer, loaders['validation'], adapters, args.seed, setup_seed, logger)
    frozen_guard(trainer, before, versions)
    if report['validation']['overall']['images'] != len(manifest['validation']):
        raise RuntimeError('Validation image count mismatch')
    report['unchanged_detector_tensors'] = len(before)
    report['unchanged_detector_tensor_versions'] = len(versions)
    with (folder / 'diagnostics/spatial_channel_pilot.json').open('x', encoding='utf8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
    logger.info('Feature validation overall: ' + json.dumps(report['validation']['overall'], allow_nan=False))
    logger.info('Spatial-channel pilot: DONE; no target evaluation; report=' + str(folder / 'diagnostics/spatial_channel_pilot.json'))
    return report
