"""Check one explicit Pack for ACT and DP without starting a training run.

Default: read only, including DP observation histories H1 and H2. Optional
--updates 1..3 exercises real model updates and an independent engineering
checkpoint under .local/checks. It does not create a deployable checkpoint or
exercise the formal trainer CLI. No Raw, Prepared, or Pack file is modified.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_config(name):
    return json.loads((ROOT / 'config' / name).read_text(encoding='utf-8'))


def inspect_sample(dataset, index, *, history=None, sample=None):
    """Check a reader sample against its original arrays and certified run."""
    key, anchor, end = dataset.windows[index]
    arrays = dataset.load_episode(key)
    info = dataset.binding['episodes'][key]
    start, run_end = next((a, b) for a, b in info['runs'] if a <= anchor < b)
    require(end == min(run_end, anchor + 30), 'Window crosses a hard boundary')
    sample = dataset[index] if sample is None else sample
    width, height = dataset.image_size
    image_shape = (1, 3, height, width) if history is None else (history, 1, 3, height, width)
    state_shape = (16,) if history is None else (history, 16)
    for name, shape in (('images', image_shape), ('state', state_shape),
                        ('action', (30, 14)), ('is_pad', (30,))):
        require(sample[name].shape == shape, f'Unexpected {name} shape')
    require(sample['is_pad'].dtype == np.bool_, 'Padding must be boolean')
    for name in ('images', 'state', 'action'):
        require(sample[name].dtype == np.float32 and np.isfinite(sample[name]).all(),
                f'{name} must be finite float32')
    require(sample['images'].min() >= 0 and sample['images'].max() <= 1, 'RGB range differs')
    ids = np.minimum(np.arange(anchor, anchor + 30), end - 1)
    padding = (np.arange(30) >= end - anchor) | ~arrays['action_mask'][ids].astype(bool)
    np.testing.assert_array_equal(sample['is_pad'], padding)
    require(not padding[0], 'A training anchor has no causal action')
    require(np.all(sample['action'][padding] == 0), 'Padding carries action targets')
    fields = {'target_raw_frame_index': 'raw_frame_index',
              'target_control_tick': 'action_control_tick',
              'target_train_sequence_id': 'train_sequence_id'}
    for output, field in fields.items():
        expected = arrays[field][ids].copy()
        expected[padding] = -1
        np.testing.assert_array_equal(sample[output], expected)
    require(np.all(arrays['action_control_tick'][ids[~padding]] ==
                   arrays['source_control_tick'][ids[~padding]] + 1),
            'Targets do not refer to the next original control tick')
    decoded = dataset.normalizer.decode_action(sample['action'])[~padding]
    delta = decoded.astype(np.float64) - arrays['action'][ids[~padding]]
    for column in (3, 4, 5, 10, 11, 12):
        delta[:, column] = (delta[:, column] + np.pi) % (2 * np.pi) - np.pi
    require(np.all(np.abs(delta) <= 2e-5), 'Action normalization round trip differs')
    if history is not None:
        positions = np.maximum(np.arange(anchor - history + 1, anchor + 1), start)
        for output, field in (('observation_raw_frame_index', 'raw_frame_index'),
                              ('observation_source_control_tick', 'source_control_tick'),
                              ('observation_train_sequence_id', 'train_sequence_id')):
            np.testing.assert_array_equal(sample[output], arrays[field][positions])
        expected_state = dataset.normalizer.encode_state(arrays['state'][positions])
    else:
        expected_state = dataset.normalizer.encode_state(arrays['state'][anchor])
    np.testing.assert_array_equal(sample['state'], expected_state)
    return {'images': list(image_shape), 'state': list(state_shape), 'action': [30, 14]}


def representative_indices(dataset):
    """Cover both sides of every certified safe seam and hard boundary."""
    chosen = set()
    first_last = {}
    by_anchor = {(key, anchor): i for i, (key, anchor, _) in enumerate(dataset.windows)}
    for i, (key, _, _) in enumerate(dataset.windows):
        first_last.setdefault(key, [i, i])[1] = i
    for first, last in first_last.values():
        chosen.update((first, last))
    for key in dataset.keys:
        info = dataset.binding['episodes'][key]
        for position in info['safe_stitch_positions'] + info['hard_boundary_positions']:
            for anchor in (position - 1, position):
                if (key, anchor) in by_anchor:
                    chosen.add(by_anchor[key, anchor])
    return sorted(chosen)


def inspect_dataset(dataset, *, history=None):
    require(len(dataset) > 0, 'Empty split')
    indices = representative_indices(dataset)
    shape = None
    for index in indices:
        shape = inspect_sample(dataset, index, history=history)
    return {'windows': len(dataset), 'checked_boundary_samples': len(indices), **shape}


def _same_state(actual, expected):
    """Exact checkpoint state comparison, independent of tensor device."""
    import torch
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=0, atol=0)
    elif isinstance(expected, dict):
        require(actual.keys() == expected.keys(), 'Checkpoint state keys changed')
        for key in expected:
            _same_state(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        require(len(actual) == len(expected), 'Checkpoint sequence length changed')
        for a, b in zip(actual, expected):
            _same_state(a, b)
    else:
        require(actual == expected, 'Checkpoint scalar changed')


def engineering_updates(algorithm, dataset, config, output, *, updates, device='cuda', batch_size=None):
    """At most three total updates, restoring after the first completed update."""
    import torch
    from torch.utils.data._utils.collate import default_collate
    from training.papera.act_model import ACTConfig, ACTPolicy, compute_loss
    from training.papera.train import (StatefulBatchSampler, atomic_checkpoint,
        lr_multiplier, move_batch, precision_context, restore_rng, rng_state, seed_everything)
    from training.diffusion.model import DiffusionPolicy
    from training.diffusion.train import EMA, learning_rate_factor, model_config
    require(type(updates) is int and 1 <= updates <= 3, 'Engineering updates must be 1..3')
    batch_size = config['batch_size'] if batch_size is None else batch_size
    require(type(batch_size) is int and batch_size >= 1, 'Batch size must be positive')
    require(algorithm in ('act', 'dp'), 'Use ACT or DP')
    output = Path(output).resolve()
    require((ROOT / '.local/checks').resolve() in output.parents, 'Engineering output must stay in .local/checks')
    require(not output.exists(), 'Use a new engineering output directory')
    require(device in ('cpu', 'cuda'), 'Use cpu or cuda')
    require(device != 'cuda' or torch.cuda.is_available(), 'CUDA is unavailable')
    precision = config['precision'] if device == 'cuda' else 'fp32'
    require(precision != 'bf16' or torch.cuda.is_bf16_supported(), 'BF16 is unavailable')
    torch.set_num_threads(config['cpu_threads'])
    device = torch.device(device)

    def construct():
        seed_everything(config['seed'])
        if algorithm == 'act':
            values = dict(config['model'], variant='baseline')
            if values.get('backbone_weights'):
                values['backbone_weights'] = str((ROOT / values['backbone_weights']).resolve())
            model = ACTPolicy(ACTConfig(**values)).to(device).train()
            backbone = {id(p) for p in model.backbone.parameters()}
            optimizer = torch.optim.AdamW([
                {'params': [p for p in model.parameters() if id(p) not in backbone], 'lr': config['learning_rate']},
                {'params': list(model.backbone.parameters()), 'lr': config['backbone_learning_rate']}],
                weight_decay=config['weight_decay'])
            multiplier, ema = lr_multiplier, None
        else:
            model = DiffusionPolicy(model_config(config)).to(device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                betas=tuple(config['betas']), weight_decay=config['weight_decay'])
            multiplier, ema = learning_rate_factor, EMA(model, config['ema_decay'])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: multiplier(step, config))
        sampler = StatefulBatchSampler(len(dataset), batch_size, config['seed'], prefetch_safe=True)
        return model, optimizer, scheduler, sampler, ema

    def update(iterator):
        indices = next(iterator)
        batch = move_batch(default_collate([dataset[i] for i in indices]), device)
        optimizer.zero_grad(set_to_none=True)
        with precision_context(device, precision):
            if algorithm == 'act':
                prediction = model(batch['images'], batch['state'], batch['action'], batch['is_pad'])
                loss = compute_loss(prediction, {'action': batch['action']},
                    {'action': ~batch['is_pad']}, config['loss_weights'], 'baseline')['total']
            else:
                loss = model.compute_loss(batch['images'], batch['state'], batch['action'], batch['is_pad'])['loss']
        require(bool(torch.isfinite(loss)), 'Nonfinite engineering loss')
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        sampler.mark_consumed()
        return {'loss': float(loss.detach().cpu()), 'gradient_norm': float(gradient.cpu()), 'indices': indices}

    def payload(step):
        value = {'version': 'papera-pack-check-engineering-v1', 'algorithm': algorithm,
            'step': step, 'pack_path': str(dataset.root), 'binding': dict(dataset.binding),
            'config': config, 'device': str(device), 'precision': precision,
            'engineering_batch_size': batch_size, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'sampler': sampler.state_dict(), 'rng': rng_state(),
            'deployable': False, 'formal_trainer_checkpoint': False}
        if ema is not None:
            value.update(ema=ema.model.state_dict(), ema_updates=ema.updates)
        return value

    output.mkdir(parents=True, exist_ok=False)
    model, optimizer, scheduler, sampler, ema = construct()
    records = [update(iter(sampler))]
    checkpoint_path = output / 'engineering.pt'
    atomic_checkpoint(checkpoint_path, payload(1))
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    require(checkpoint['version'] == 'papera-pack-check-engineering-v1' and
            checkpoint['algorithm'] == algorithm and checkpoint['config'] == config and
            checkpoint['binding'] == dict(dataset.binding),
            'Engineering checkpoint data/config/normalization binding changed')
    del model, optimizer, scheduler, sampler, ema
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    model, optimizer, scheduler, sampler, ema = construct()
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    sampler.load_state_dict(checkpoint['sampler'])
    if ema is not None:
        ema.model.load_state_dict(checkpoint['ema'])
        ema.updates = checkpoint['ema_updates']
        _same_state(ema.model.state_dict(), checkpoint['ema'])
    restore_rng(checkpoint['rng'])
    for actual, saved in ((model.state_dict(), checkpoint['model']),
                          (optimizer.state_dict(), checkpoint['optimizer']),
                          (scheduler.state_dict(), checkpoint['scheduler']),
                          (sampler.state_dict(), checkpoint['sampler']), (rng_state(), checkpoint['rng'])):
        _same_state(actual, saved)
    del checkpoint
    iterator = iter(sampler)
    for _ in range(1, updates):
        records.append(update(iterator))
    atomic_checkpoint(checkpoint_path, payload(updates))
    result = {'updates': updates, 'resumed_updates': updates - 1, 'checkpoint_restore_equal': True,
        'batch_size': batch_size, 'precision': precision, 'device': str(device),
        'observation_horizon': 1 if algorithm == 'act' else config['model']['observation_horizon'],
        'checkpoint': str(checkpoint_path), 'records': records, 'deployable': False,
        'trajectory_equivalence_tested': False,
        'scope': 'tensor_updates_and_state_restore_only_not_formal_trainer_cli'}
    (output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result


def check_pack(pack, task_id, *, full_verify=False, act_config=None, dp_config=None,
               updates=0, output=None, device='cuda', batch_size=None):
    from dataset.task3 import training_pack
    from training.papera.pack_data import prepare_pack, SafeStitchDataset
    from training.diffusion.data import prepare_data, DiffusionDataset
    require(type(updates) is int and 0 <= updates <= 3, 'Engineering updates must be 0..3')
    require(updates > 0 or output is None, 'Read-only check does not write an output directory')
    act_config = copy.deepcopy(act_config if act_config is not None else load_config('training_baseline.json'))
    dp_config = copy.deepcopy(dp_config if dp_config is not None else load_config('training_diffusion.json'))
    require(act_config['image_size'] == dp_config['image_size'], 'ACT/DP image sizes must match')
    for config in (act_config, dp_config):
        require(config['model']['state_dim'] == 16 and config['model']['action_dim'] == 14, 'Expected S16/A14')
    require(act_config['model']['num_cameras'] == 1 and act_config['model']['chunk_size'] == 30, 'Expected single view/K30')
    require(dp_config['model']['prediction_horizon'] == 30 and dp_config['model']['observation_horizon'] in (1, 2),
            'Expected DP K30 and H1 or H2')
    checked = training_pack.verify(Path(pack).resolve(), check_hashes=full_verify)
    manifest = checked['manifest']
    require(manifest['task_id'] == task_id, 'Cross-task Pack input rejected')
    engineering = bool(manifest['engineering_only'])
    act_binding = prepare_pack(pack, checked=checked, image_size=act_config['image_size'])
    act = {side: SafeStitchDataset(act_binding, split=side) for side in ('train', 'val')}
    readers = list(act.values())
    report = {'task_id': task_id, 'pack_path': str(Path(pack).resolve()),
        'episodes': manifest['episode_count'], 'frames': manifest['frame_count'],
        'engineering_data': engineering, 'verification': 'full_hashes' if full_verify else 'format_and_interfaces',
        'contract': {'state_dim': 16, 'action_dim': 14, 'chunk_size': 30,
                     'image_size': act_config['image_size'], 'cameras': ['oblique'], 'action_hz': 30},
        'configured_models': {'act': act_config['model'], 'dp': dp_config['model']},
        'counts': act_binding['counts'], 'act': {}, 'dp': {}, 'training_started': bool(updates),
        'formal_training_started': False, 'task_success_claimed': False}
    output = (ROOT / '.local/checks/training-pack' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')) if output is None else Path(output)
    try:
        if updates:
            output = output.resolve()
            require((ROOT / '.local/checks').resolve() in output.parents, 'Engineering output must stay in .local/checks')
            require(not output.exists(), 'Use a new engineering output directory')
        require(act_binding['stats']['fit_split'] == 'train' and
                act_binding['stats']['training_keys'] == act_binding['train_keys'], 'ACT normalization leaks split')
        for side, dataset in act.items():
            report['act'][side] = inspect_dataset(dataset)
        for history in (1, 2):
            binding = prepare_data(pack, task_id, {**dp_config['model'], 'image_size': dp_config['image_size'],
                'observation_horizon': history}, check_hashes=full_verify, engineering=engineering)
            require(binding['split'] == act_binding['split'] and binding['episodes'] == act_binding['episodes'],
                    'ACT/DP splits or certified windows differ')
            require(binding['stats']['fit_split'] == 'train' and
                    binding['stats']['training_keys'] == binding['train_keys'], 'DP normalization leaks split')
            report['dp'][f'H{history}'] = {}
            for side in ('train', 'val'):
                dataset = DiffusionDataset(binding, split=side)
                readers.append(dataset)
                require(dataset.windows == act[side].windows, 'ACT/DP window order differs')
                report['dp'][f'H{history}'][side] = inspect_dataset(dataset, history=history)
                if updates and side == 'train' and history == dp_config['model']['observation_horizon']:
                    report['dp']['engineering'] = engineering_updates('dp', dataset, dp_config,
                        output / 'dp', updates=updates, device=device, batch_size=batch_size)
        if updates:
            report['act']['engineering'] = engineering_updates('act', act['train'], act_config,
                output / 'act', updates=updates, device=device, batch_size=batch_size)
            report['engineering_updates_started'] = True
            (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        return report
    finally:
        for dataset in readers:
            dataset.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--full-verify', action='store_true', help='Hash every Pack file; default checks formats and evidence')
    parser.add_argument('--updates', type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument('--output', type=Path, help='New .local/checks directory; requires --updates')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--batch-size', type=int, help='Explicit engineering batch override; default reads each JSON batch_size')
    args = parser.parse_args(argv)
    report = check_pack(args.pack, args.task, full_verify=args.full_verify, updates=args.updates,
                        output=args.output, device=args.device, batch_size=args.batch_size)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
