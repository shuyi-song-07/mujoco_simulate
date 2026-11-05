"""Load a trained DP's EMA weights without requiring its pretrained-weight file.

This is a tensor inference interface, not an adapter to the ACT rollout CLI.
The caller supplies RGB/state history in the exact frozen preprocessing format.
"""
from pathlib import Path

import torch

from training.diffusion.model import DiffusionConfig, DiffusionPolicy
from training.diffusion.data import Normalizer
from training.diffusion.train import CHECKPOINT_VERSION


def load_policy(checkpoint, device='cpu'):
    """Return (EMA policy, normalizer, frozen metadata), without opening Raw data."""
    saved = torch.load(Path(checkpoint), map_location='cpu', weights_only=False)
    if saved.get('version') == 'papera-diffusion-checkpoint-v1':
        raise ValueError('Legacy local DP checkpoint cannot load into the official-core network; start a new training run')
    if saved.get('version') != CHECKPOINT_VERSION:
        raise ValueError('Expected a Paper A Diffusion Policy checkpoint')
    config = dict(saved['model_config'])
    config['backbone_weights'] = None  # A full checkpoint already contains the backbone.
    config['down_dims'] = tuple(config['down_dims'])
    model = DiffusionPolicy(DiffusionConfig(**config)).to(device)
    model.load_state_dict(saved['ema'])
    model.eval()
    normalizer = Normalizer(saved['data_binding']['stats'])
    metadata = {key: saved[key] for key in ('task_id', 'pack_path', 'model_config', 'step', 'engineering')}
    metadata['image_size'] = saved['config']['image_size']
    metadata['sampler'] = 'official_diffusers_DDPM'
    metadata['image_crop_shape'] = list(model.config.crop_shape) if model.config.crop_shape is not None else None
    metadata['action_alignment'] = 'first_prediction_is_next_legal_control_tick_after_latest_observation'
    return model, normalizer, metadata
