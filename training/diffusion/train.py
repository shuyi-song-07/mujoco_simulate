"""Train a visual Diffusion Policy from the project's formal Training Pack.

No raw-data reconstruction or second Safe Stitch decision occurs here.
Default validation checks formats and causal interfaces; --full-verify also
hashes the Pack. ACT, teleoperation and Dataset curation are independent.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.papera.train import (StatefulBatchSampler, atomic_checkpoint,
    move_batch, precision_context, rng_state, restore_rng, reconcile_resume_logs)
from training.diffusion.model import DiffusionConfig, DiffusionPolicy
from training.diffusion.data import DiffusionDataset, prepare_data, restore_data

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_VERSION = 'papera-diffusion-checkpoint-v1'


def resolve(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def validate_config(config):
    for name in ('steps', 'batch_size', 'cpu_threads', 'validation_every', 'save_every', 'log_every'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    for name in ('num_workers', 'warmup_steps', 'validation_sample_batches'):
        if type(config[name]) is not int or config[name] < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    limit = config['validation_batches']
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('validation_batches must be null (all) or a positive integer')
    if limit is not None and config['validation_sample_batches'] > limit:
        raise ValueError('validation_sample_batches cannot exceed validation_batches')
    for name in ('learning_rate', 'gradient_clip_norm'):
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if not math.isfinite(config['weight_decay']) or config['weight_decay'] < 0:
        raise ValueError('weight_decay must be finite and nonnegative')
    for name in ('ema_decay', 'minimum_lr_fraction'):
        if not math.isfinite(config[name]) or not 0 <= config[name] < 1:
            raise ValueError(f'{name} must be finite in [0,1)')
    if len(config['betas']) != 2 or any(not math.isfinite(b) or not 0 <= b < 1 for b in config['betas']):
        raise ValueError('AdamW betas must contain two values in [0,1)')
    if config['precision'] not in ('fp32', 'bf16') or config['device'] not in ('cpu', 'cuda'):
        raise ValueError('Use device=cpu|cuda and precision=fp32|bf16')
    if len(config['image_size']) != 2 or any(type(v) is not int or v < 32 for v in config['image_size']):
        raise ValueError('image_size must be [width,height], each >=32')
    return config


class EMA:
    """Warm-started exponential moving average, including saved update count."""
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = float(decay)
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for average, value in zip(self.model.parameters(), model.parameters()):
            average.lerp_(value.detach(), 1 - decay)
        for average, value in zip(self.model.buffers(), model.buffers()):
            average.copy_(value)


def seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def learning_rate_factor(step, config):
    if step < config['warmup_steps']:
        return (step + 1) / max(1, config['warmup_steps'])
    progress = (step - config['warmup_steps']) / max(1, config['steps'] - config['warmup_steps'])
    return config['minimum_lr_fraction'] + (1 - config['minimum_lr_fraction']) * (
        1 + math.cos(math.pi * min(1., max(0., progress)))) / 2


@torch.no_grad()
def validate(model, loader, normalizer, device, config):
    """Fixed noise/timesteps make repeated EMA validation comparable."""
    model.eval()
    noise_rng = torch.Generator(device=device).manual_seed(config['seed'] + 100)
    sample_rng = torch.Generator(device=device).manual_seed(config['seed'] + 200)
    squared_error, valid_elements = 0., 0
    absolute_error, sample_elements = 0., 0
    sampled_physical = {'position_abs_m': 0., 'rotation_abs_rad': 0., 'aperture_abs_m': 0.}
    physical_counts = dict.fromkeys(sampled_physical, 0)
    batches = 0
    for i, batch in enumerate(loader):
        if config['validation_batches'] is not None and i >= config['validation_batches']:
            break
        b = move_batch(batch, device)
        count = int((~b['is_pad']).sum()) * 14
        with precision_context(device, config['precision']):
            losses = model.compute_loss(b['images'], b['state'], b['action'], b['is_pad'], generator=noise_rng)
            if i < config['validation_sample_batches']:
                prediction = model.predict_action(b['images'], b['state'], generator=sample_rng)
        squared_error += float(losses['noise_mse']) * count
        valid_elements += count
        batches += 1
        if i < config['validation_sample_batches']:
            mask = (~b['is_pad'])[..., None].expand_as(prediction)
            absolute_error += float((prediction.float() - b['action']).abs()[mask].sum())
            sample_elements += int(mask.sum())
            pred = normalizer.decode_action(prediction.float().cpu().numpy())
            target = normalizer.decode_action(b['action'].float().cpu().numpy())
            valid = ~b['is_pad'].cpu().numpy()
            for name, columns in (('position_abs_m', [0,1,2,7,8,9]),
                                  ('rotation_abs_rad', [3,4,5,10,11,12]),
                                  ('aperture_abs_m', [6,13])):
                error = pred[..., columns] - target[..., columns]
                if name == 'rotation_abs_rad':
                    error = (error + np.pi) % (2 * np.pi) - np.pi
                sampled_physical[name] += float(np.abs(error)[valid].sum())
                physical_counts[name] += int(valid.sum()) * len(columns)
    if not valid_elements:
        raise ValueError('Validation has no valid action targets')
    result = {'noise_mse': squared_error / valid_elements, 'batches': batches,
              'valid_action_elements': valid_elements, 'ema': True,
              'validation_batch_limit': config['validation_batches']}
    if sample_elements:
        result['sample_action_l1_normalized'] = absolute_error / sample_elements
        result.update({key: value / physical_counts[key] for key, value in sampled_physical.items()})
    if not all(math.isfinite(value) for value in result.values() if isinstance(value, float)):
        raise FloatingPointError('Nonfinite validation metric')
    return result


def model_config(config):
    values = dict(config['model'])
    if values.get('backbone_weights'):
        values['backbone_weights'] = str(resolve(values['backbone_weights']))
    values['down_dims'] = tuple(values['down_dims'])
    return DiffusionConfig(**values)


def data_config(config):
    return {**config['model'], 'image_size': config['image_size']}


def append_metrics(output, writer, event, step, values):
    record = {'event': event, 'step': step, **values}
    with (output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
    for key, value in values.items():
        if isinstance(value, (float, int)):
            writer.add_scalar(f'{event}/{key}', value, step)


def run(config, pack, task_id, *, output=None, resume=None, engineering=False,
        full_verify=False, stop_after=None, check_only=False):
    config = copy.deepcopy(validate_config(config))
    torch.set_num_threads(config['cpu_threads'])
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError('stop_after must be a positive number of updates')
    pack = resolve(pack)
    checkpoint = None
    if resume:
        checkpoint = torch.load(resolve(resume), map_location='cpu', weights_only=False)
        if checkpoint.get('version') != CHECKPOINT_VERSION or checkpoint.get('task_id') != task_id:
            raise ValueError('Checkpoint is not a compatible Diffusion Policy for this task')
        if (checkpoint['config'] != config or Path(checkpoint['pack_path']) != pack
                or bool(checkpoint['engineering']) != bool(engineering)):
            raise ValueError('Resume requires the frozen config, Pack path and data scope')
        binding = restore_data(checkpoint['data_binding'], check_hashes=full_verify)
    else:
        binding = prepare_data(pack, task_id, data_config(config),
                               check_hashes=full_verify, engineering=engineering)
    train = DiffusionDataset(binding, split='train')
    heldout = DiffusionDataset(binding, split='val')
    try:
        first = train[0]
        contract = {'task_id': task_id, 'pack_path': str(pack),
                    'images': list(first['images'].shape), 'state': list(first['state'].shape),
                    'action': list(first['action'].shape), 'padding': list(first['is_pad'].shape),
                    'train_windows': len(train), 'validation_windows': len(heldout),
                    'validation_level': 'full_hashes' if full_verify else 'format_and_interfaces'}
        if check_only:
            print(json.dumps(contract, ensure_ascii=False, indent=2), flush=True)
            return contract
        device = torch.device(config['device'])
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; select device=cpu in an engineering config')
        if device.type == 'cuda' and config['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 is unavailable; use precision=fp32')
        seed(config['seed'])
        mc = model_config(config)
        model = DiffusionPolicy(mc).to(device)
        ema = EMA(model, config['ema_decay'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
            betas=tuple(config['betas']), weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: learning_rate_factor(t, config))
        sampler = StatefulBatchSampler(len(train), config['batch_size'], config['seed'], prefetch_safe=True)
        loader_rng = torch.Generator().manual_seed(config['seed'] + 1)
        options = {'num_workers': config['num_workers'], 'pin_memory': device.type == 'cuda', 'generator': loader_rng}
        if config['num_workers']:
            options.update(multiprocessing_context='spawn', persistent_workers=True, prefetch_factor=2)
        train_loader = DataLoader(train, batch_sampler=sampler, **options)
        val_loader = DataLoader(heldout, batch_size=config['batch_size'], shuffle=False,
                               num_workers=0, pin_memory=device.type == 'cuda',
                               generator=torch.Generator().manual_seed(config['seed'] + 2))
        step, best, elapsed_offset = 0, math.inf, 0.
        if resume:
            model.load_state_dict(checkpoint['model'])
            ema.model.load_state_dict(checkpoint['ema'])
            ema.updates = checkpoint['ema_updates']
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            sampler.load_state_dict(checkpoint['sampler'])
            loader_rng.set_state(checkpoint['loader_generator'].cpu())
            step, best = checkpoint['step'], checkpoint['best_noise_mse']
            elapsed_offset = checkpoint.get('elapsed_s', 0.)
            output = resolve(resume).parent.parent
            reconcile_resume_logs(output, step)
        else:
            output = resolve(output) if output else resolve(config['output_root']) / task_id / 'diffusion' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            if engineering and not output.resolve().is_relative_to((ROOT / '.local/checks').resolve()):
                if output.is_relative_to(resolve(config['output_root'])):
                    output = ROOT / '.local/checks/diffusion' / output.name
                else:
                    raise ValueError('Engineering outputs must be inside .local/checks')
            output.mkdir(parents=True, exist_ok=False)
            (output / 'checkpoints').mkdir()
            write_json(output / 'config.json', config)
            write_json(output / 'data_binding.json', dict(binding))
            write_json(output / 'run.json', {'algorithm': 'diffusion_policy', 'version': CHECKPOINT_VERSION,
                'task_id': task_id, 'pack_path': str(pack), 'engineering': engineering,
                'model_config': asdict(mc), 'config': config, 'contract': contract,
                'runtime': {'python': sys.version, 'executable': sys.executable,
                    'torch': str(torch.__version__), 'device': str(device),
                    'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None},
                'best_selection': 'EMA validation noise MSE with fixed random draws',
                'closed_loop_verified': False})
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(output / 'tensorboard'), purge_step=step if resume else None)
        print(f'Diffusion Policy | train={len(train)}, val={len(heldout)} windows | output={output}', flush=True)
        steps_per_epoch = math.ceil(len(train) / config['batch_size'])
        print(f"预算 {config['steps']} 次更新，约 {config['steps']/steps_per_epoch:.2f} 轮；每轮 {steps_per_epoch} 步。", flush=True)
        print('best 按 EMA 验证噪声 MSE 选择；离线误差不是任务成功率。Ctrl+C 保存 last.pt。', flush=True)
        iterator = iter(train_loader)
        if checkpoint:
            # Iterator construction consumes a worker seed; restore training RNG afterwards.
            loader_rng.set_state(checkpoint['loader_generator'].cpu())
            restore_rng(checkpoint['rng'])
            del checkpoint
        start = time.perf_counter()
        start_step = step
        status = 'failed'
        stop_requested = False
        previous_handler = None

        def request_stop(signum, frame):
            nonlocal stop_requested
            stop_requested = True

        # Defer Ctrl+C until a complete optimizer/sampler update. A checkpoint
        # cannot represent half an update or an already-prefetched batch.
        import threading
        if threading.current_thread() is threading.main_thread():
            previous_handler = signal.signal(signal.SIGINT, request_stop)

        def save(name):
            atomic_checkpoint(output / 'checkpoints' / name, {
                'version': CHECKPOINT_VERSION, 'algorithm': 'diffusion_policy',
                'task_id': task_id, 'pack_path': str(pack), 'config': config,
                'data_binding': dict(binding), 'model_config': asdict(mc),
                'engineering': engineering, 'step': step, 'best_noise_mse': best,
                'model': model.state_dict(), 'ema': ema.model.state_dict(), 'ema_updates': ema.updates,
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'sampler': sampler.state_dict(), 'rng': rng_state(),
                'loader_generator': loader_rng.get_state(),
                'elapsed_s': elapsed_offset + time.perf_counter() - start,
                'closed_loop_verified': False})

        try:
            model.train()
            while step < config['steps']:
                batch = move_batch(next(iterator), device)
                optimizer.zero_grad(set_to_none=True)
                with precision_context(device, config['precision']):
                    losses = model.compute_loss(batch['images'], batch['state'], batch['action'], batch['is_pad'])
                loss = losses['loss']
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite diffusion loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                ema.update(model)
                sampler.mark_consumed()
                step += 1
                if step == 1 or step % config['log_every'] == 0:
                    values = {'noise_mse': float(losses['noise_mse'].detach()),
                              'lr': optimizer.param_groups[0]['lr'], 'gradient_norm': float(norm),
                              'epoch': sampler.state_dict()['epoch'],
                              'elapsed_s': elapsed_offset + time.perf_counter() - start}
                    append_metrics(output, writer, 'train', step, values)
                    print(f"step {step}/{config['steps']} noise_mse={values['noise_mse']:.6f}", flush=True)
                final_update = (step == config['steps'] or stop_requested
                                or (stop_after is not None and step - start_step >= stop_after))
                if step % config['validation_every'] == 0 or final_update:
                    metrics = validate(ema.model, val_loader, train.normalizer, device, config)
                    append_metrics(output, writer, 'validation', step, metrics)
                    if metrics['noise_mse'] < best:
                        best = metrics['noise_mse']
                        save('best.pt')
                    print(f"validation step={step} EMA noise_mse={metrics['noise_mse']:.6f} best={best:.6f}", flush=True)
                if step % config['save_every'] == 0 or final_update:
                    save('last.pt')
                if final_update:
                    break
            status = ('complete' if step >= config['steps'] else 'interrupted' if stop_requested
                      else 'stopped_after_requested_updates')
            if stop_requested:
                print('已在完整更新后保存 last.pt，可使用 --resume 继续。', flush=True)
        finally:
            if previous_handler is not None:
                signal.signal(signal.SIGINT, previous_handler)
            writer.close()
            write_json(output / 'completion.json', {'status': status, 'step': step,
                'budget_steps': config['steps'], 'best_noise_mse': best if math.isfinite(best) else None,
                'elapsed_s': elapsed_offset + time.perf_counter() - start,
                'engineering': engineering, 'closed_loop_verified': False})
            # DataLoader workers hold their own memory maps on Windows.
            if getattr(train_loader, '_iterator', None) is not None:
                train_loader._iterator._shutdown_workers()
        return output
    finally:
        train.close()
        heldout.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--variant', choices=('baseline', 'gesture'), default='baseline',
                        help='baseline 已实现；gesture 为后续手势辅助监督实验预留')
    parser.add_argument('--pack', type=Path, help='Explicit formal Training Pack directory; never a Raw/merged export')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--steps', type=int, help='Total optimizer updates for a new run')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path, help='Exact checkpoints/last.pt; restores frozen config')
    parser.add_argument('--engineering', action='store_true', help='Read explicitly scripted engineering Pack')
    parser.add_argument('--full-verify', action='store_true', help='Also verify file hashes; slower, optional')
    parser.add_argument('--check-only', action='store_true', help='Check data tensors and sequence interfaces; no training')
    parser.add_argument('--stop-after', type=int, help='Stop after N additional updates and save; budget unchanged')
    args = parser.parse_args(argv)
    if args.variant == 'gesture':
        parser.error('DP + 手势辅助监督尚未实现：目前只有 DP Baseline。该入口不会启动训练，也不会替换为 Baseline。')
    if args.resume:
        if args.output or args.steps is not None or args.config:
            parser.error('--resume reads its frozen config; do not combine with --output/--steps/--config')
        if args.resume.name != 'last.pt':
            parser.error('Resume from checkpoints/last.pt; best.pt is intended for inference')
        saved = json.loads((resolve(args.resume).parent.parent / 'run.json').read_text(encoding='utf-8'))
        if saved.get('version') != CHECKPOINT_VERSION:
            parser.error('Not a Diffusion Policy checkpoint')
        config, pack = saved['config'], Path(saved['pack_path'])
        if args.pack and resolve(args.pack) != pack:
            parser.error('--pack differs from the checkpoint')
        engineering = bool(saved['engineering'])
        del saved
    else:
        if args.pack is None:
            parser.error('--pack is required for a new run')
        config = json.loads(resolve(args.config or 'config/training_diffusion.json').read_text(encoding='utf-8-sig'))
        if args.steps is not None:
            config['steps'] = args.steps
        pack, engineering = args.pack, args.engineering
    result = run(config, pack, args.task, output=args.output, resume=args.resume,
                 engineering=engineering, full_verify=args.full_verify,
                 stop_after=args.stop_after, check_only=args.check_only)
    if isinstance(result, Path):
        print('Output: ' + str(result), flush=True)


if __name__ == '__main__':
    main()
