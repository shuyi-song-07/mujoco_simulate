"""EMA Diffusion Policy adapter for the shared MuJoCo rollout executor.

History advances on every 30 Hz observation, including non-query steps. Future
action zero retains the Pack's next-control-tick meaning; no To-1 shift is used.
"""
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from training.diffusion.data import Normalizer, VERSION as DATA_VERSION
from training.diffusion.model import DiffusionConfig, DiffusionPolicy
from training.diffusion.train import CHECKPOINT_VERSION, model_config
from training.papera.data import file_sha256
from training.papera.pack_data import resize_rgb
from training.papera.train import precision_context


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def inference_metadata(metadata, settings):
    """Return audited inference overrides without rewriting frozen checkpoints."""
    result = dict(metadata)
    frozen_steps = metadata.get('training_inference_steps', metadata['inference_steps'])
    frozen_horizon = metadata.get('training_action_horizon', metadata['action_horizon'])
    steps = settings.get('diffusion_inference_steps', frozen_steps)
    horizon = settings.get('diffusion_action_horizon', frozen_horizon)
    if type(steps) is not int or not 1 <= steps <= metadata['diffusion_steps']:
        raise ValueError('diffusion_inference_steps must be within the trained diffusion schedule')
    if type(horizon) is not int or not 1 <= horizon <= metadata['chunk_size']:
        raise ValueError('diffusion_action_horizon must be within the prediction horizon')
    changed = steps != frozen_steps or horizon != frozen_horizon
    reason = settings.get('inference_selection_reason')
    if changed and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError('DP inference overrides require inference_selection_reason')
    result.update(training_inference_steps=frozen_steps, training_action_horizon=frozen_horizon,
        inference_steps=steps, action_horizon=horizon, inference_overridden=changed,
        inference_selection_reason=reason if changed else 'frozen training defaults')
    return result


def verify_run(run_dir, checkpoint='best', *, verify_cache=False, allow_engineering=False):
    """Verify the frozen DP run/selection/interfaces, without opening raw RGB."""
    from dataset.task3 import training_pack
    from tasks.artifacts import check_artifact
    from rollout.protocol import load_scene_provenance
    if checkpoint not in ('best', 'last'):
        raise ValueError('Select best or last')
    root = Path(run_dir).resolve()
    run, completion = _read(root/'run.json'), _read(root/'completion.json')
    config, binding = _read(root/'config.json'), _read(root/'data_binding.json')
    engineering = run.get('engineering')
    if (run.get('algorithm') != 'diffusion_policy' or run.get('version') != CHECKPOINT_VERSION
            or run.get('config') != config or type(engineering) is not bool
            or completion.get('engineering') != engineering):
        raise ValueError('Incompatible Diffusion Policy run/config/scope')
    if engineering and not allow_engineering:
        raise ValueError('Engineering DP requires --engineering and cannot enter formal batch evaluation')
    if (completion.get('status') != 'complete' or completion.get('step') != config['steps']
            or completion.get('budget_steps') != config['steps'] or config['steps'] <= 0):
        raise ValueError('DP evaluation requires a completed training budget')
    pack = Path(run['pack_path']).resolve()
    manifest = check_artifact(pack, run['task_id'], engineering=engineering)
    if (binding.get('version') != DATA_VERSION or binding.get('task_id') != run['task_id']
            or binding.get('task_binding') != manifest['task_binding']
            or Path(binding['pack_path']).resolve() != pack
            or binding.get('split') != manifest['split']
            or binding.get('engineering_only') != engineering
            or binding.get('cameras') != ['oblique']
            or manifest.get('complete') is not True or manifest.get('state_dim') != 16
            or manifest.get('action_dim') != 14 or manifest.get('cameras') != ['oblique']):
        raise ValueError('DP Pack task/schema/split differs from its frozen run')
    from dataset.task3.prepare_paper_dataset import digest
    if binding.get('manifest_digest') != digest(manifest):
        raise ValueError('DP Pack manifest changed since training')
    split, stats = binding['split'], binding['stats']
    if (set(split['train']) & set(split['val']) or set(split['train_sessions']) & set(split['val_sessions'])
            or binding['train_keys'] != split['train']
            or stats.get('training_keys') != binding['train_keys']):
        raise ValueError('DP normalization or training/validation membership differs')
    normalizer = Normalizer(stats)  # Validate all dimensions, finite scales and min-max semantics.
    mc = model_config(config)
    # v1 baseline checkpoints predate the optional gesture fields. Only the
    # dataclass defaults may fill missing fields; frozen values still compare.
    frozen_mc = DiffusionConfig(**run['model_config'])
    if asdict(frozen_mc) != asdict(mc):
        raise ValueError('DP frozen model differs from its config')
    if (binding['config']['observation_horizon'] != mc.observation_horizon
            or binding['config']['prediction_horizon'] != mc.prediction_horizon
            or binding['config']['action_horizon'] != mc.action_horizon
            or binding['config']['image_size'] != config['image_size']):
        raise ValueError('DP model/data history or horizon mismatch')
    checked = training_pack.verify(pack, check_hashes=False) if verify_cache else None
    path = root/'checkpoints'/f'{checkpoint}.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    for key, value in (('version', CHECKPOINT_VERSION), ('algorithm', 'diffusion_policy'),
                       ('task_id', run['task_id']), ('pack_path', run['pack_path']),
                       ('config', config), ('data_binding', binding),
                       ('model_config', run['model_config']), ('engineering', engineering)):
        # Dataclass tuples become JSON lists in run.json; checkpoint retains tuples.
        actual = saved.get(key)
        if key == 'model_config':
            actual = json.loads(json.dumps(actual))
        if actual != value:
            raise ValueError(f'DP checkpoint differs from frozen {key}')
    validations = [json.loads(line) for line in (root/'metrics.jsonl').read_text(encoding='utf-8').splitlines()
                   if line.strip() and json.loads(line).get('event') == 'validation']
    if not validations or any(not np.isfinite(row['noise_mse']) or row.get('ema') is not True for row in validations):
        raise ValueError('DP best selection needs finite EMA validation evidence')
    best = min(validations, key=lambda row: row['noise_mse'])
    if (saved.get('step') != (best['step'] if checkpoint == 'best' else completion['step'])
            or not np.isclose(completion['best_noise_mse'], best['noise_mse'], rtol=0, atol=1e-12)):
        raise ValueError('DP selected checkpoint does not match validation noise MSE selection')
    if not saved.get('ema') or any(not torch.isfinite(v).all().item()
            for v in saved['ema'].values() if torch.is_tensor(v) and v.is_floating_point()):
        raise ValueError('DP EMA weights are missing/nonfinite')
    scene = load_scene_provenance(pack)
    h = mc.observation_horizon
    manifest_sha = file_sha256(pack/'manifest.json')
    metadata = {'algorithm': 'diffusion_policy', 'version': 'papera-diffusion-rollout-v1',
        'run_dir': str(root), 'variant': 'diffusion_gesture' if mc.variant == 'gesture' else 'diffusion',
        'training_variant': mc.variant, 'checkpoint': str(path),
        'checkpoint_sha256': file_sha256(path), 'checkpoint_step': saved['step'],
        'selected_by': 'validation/EMA_noise_mse' if checkpoint == 'best' else 'training/completed_step', 'weights': 'ema',
        'pack': str(pack), 'checked_hashes': {str(pack/'manifest.json'): manifest_sha},
        'verified_cache_files': sum(len(ep['files']) for ep in manifest['episodes']) if checked else 0,
        'full_cache_verified': False, 'verification': 'format_and_interfaces',
        'training_precision': config['precision'], 'cameras': ['oblique'],
        'training_camera_configs': {key: value['camera'] for key, value in scene['oblique_render_sources'].items()},
        'chunk_size': mc.prediction_horizon, 'action_horizon': mc.action_horizon,
        'inference_steps': mc.inference_steps, 'diffusion_steps': mc.diffusion_steps,
        'observation_horizon': h, 'model_image_shape': [h, 1, 3, *reversed(config['image_size'])],
        'source_image_shape': [480, 640, 3], 'image_size': config['image_size'],
        'data_contract': DATA_VERSION, 'action_alignment': manifest['action_alignment'],
        'human_inference_input': False, 'engineering_overfit': engineering,
        'scope': 'user-authorized standalone MuJoCo evaluation; not hardware deployment'}
    return metadata, saved, normalizer


class DiffusionRolloutPolicy:
    def __init__(self, run_dir, checkpoint='best', device='cuda', *, verify_cache=False, allow_engineering=False):
        self.metadata, saved, self.normalizer = verify_run(run_dir, checkpoint,
            verify_cache=verify_cache, allow_engineering=allow_engineering)
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Use explicit cuda or cpu engineering inference')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; DP evaluation never silently falls back to CPU')
        self.precision = self.metadata['training_precision'] if self.device.type == 'cuda' else 'fp32'
        if self.precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('Original BF16 inference is unavailable')
        fields = dict(saved['model_config'], backbone_weights=None)
        fields['down_dims'] = tuple(fields['down_dims'])
        self.model = DiffusionPolicy(DiffusionConfig(**fields)).to(self.device)
        self.model.load_state_dict(saved['ema'], strict=True)
        self.model.requires_grad_(False).eval()
        self.history = deque(maxlen=self.metadata['observation_horizon'])
        self.metadata.update(device=str(self.device), inference_precision=self.precision,
                             model_eval=True, gradients=False, history_clock_hz=30)
        self.reset(0)

    def configure_inference(self, settings):
        self.metadata = inference_metadata(self.metadata, settings)

    def reset(self, seed):
        self.history.clear()
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))

    def observe(self, images, state):
        if set(images) != {'oblique'}:
            raise ValueError('DP needs exactly one oblique RGB camera')
        image, state = np.asarray(images['oblique']), np.asarray(state)
        if (image.shape != (480,640,3) or image.dtype != np.uint8 or state.shape != (16,)
                or not np.isfinite(state).all()):
            raise ValueError('DP observation must be RGB uint8 480x640 plus finite S16')
        image = resize_rgb(image, self.metadata['image_size']).transpose(2,0,1)[None].astype(np.float32)/255.
        observation = (image, self.normalizer.encode_state(state))
        if not self.history:
            for _ in range(self.history.maxlen):
                self.history.append(observation)
        else:
            self.history.append(observation)

    @torch.inference_mode()
    def predict(self, images, state):
        # Shared rollout calls observe on every frame before the chunk scheduler.
        if not self.history:
            raise RuntimeError('DP history was not advanced on the observation clock')
        rgb = torch.as_tensor(np.stack([x[0] for x in self.history]), device=self.device).unsqueeze(0)
        states = torch.as_tensor(np.stack([x[1] for x in self.history]), device=self.device).unsqueeze(0)
        with precision_context(self.device, self.precision):
            options = ({'num_inference_steps': self.metadata['inference_steps']}
                       if self.metadata['inference_steps'] != self.model.config.inference_steps else {})
            output = self.model.predict_action(rgb, states, generator=self.generator, **options)
        prediction = output.float().cpu().numpy()[0]
        if prediction.shape != (self.metadata['chunk_size'],14) or not np.isfinite(prediction).all():
            from rollout.policy import InvalidPolicyPrediction
            raise InvalidPolicyPrediction('DP returned a malformed/nonfinite action chunk')
        action = self.normalizer.decode_action(prediction)
        if not np.isfinite(action).all():
            from rollout.policy import InvalidPolicyPrediction
            raise InvalidPolicyPrediction('DP decoded nonfinite robot targets')
        return action
