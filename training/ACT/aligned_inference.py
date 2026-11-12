"""Load only versioned single-arm GJB checkpoints; old LeRobot loads stay separate."""
from collections import deque
from types import SimpleNamespace
import numpy as np
import torch
from .aligned_model import ACTConfig, ACTPolicy
from .aligned_data import Normalizer
from .aligned_training import VERSION


class SingleArmPolicy:
    def __init__(self, checkpoint, device, action_steps):
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if payload.get('version')!=VERSION:raise ValueError('Not a single-arm GJB-aligned checkpoint')
        fields=payload['config']['model']
        if action_steps is None:action_steps=fields['chunk_size']
        if not 1<=action_steps<=fields['chunk_size']:raise ValueError('Action steps exceed checkpoint chunk size')
        if payload['provenance']['image_key']!='observation.images.overview':raise ValueError('Wrong camera')
        self.model=ACTPolicy(ACTConfig(**fields)).eval().to(device)
        self.model.load_state_dict(payload['model'],strict=True)
        self.normalizer=Normalizer(payload['normalization'])
        self.device=device;self.queue=deque();self.action_steps=action_steps
        self.config=SimpleNamespace(chunk_size=fields['chunk_size'],n_action_steps=action_steps)
        self.image_shape=tuple(payload['provenance']['image_shape'])

    def reset(self):self.queue.clear()

    @torch.inference_mode()
    def select_action(self, observation):
        if not self.queue:
            image=observation['observation.images.overview']
            if image.ndim==3:image=image.unsqueeze(0)
            if tuple(image.shape[-2:])!=self.image_shape[:2]:
                raise ValueError('Rollout camera resolution differs from training')
            state=observation['observation.state'].detach().cpu().numpy().reshape(1,9)
            state=torch.from_numpy(self.normalizer.encode_state(state)).to(self.device)
            predictions=self.model(image.to(self.device).unsqueeze(1),state)['action'][0].float().cpu().numpy()
            actions=self.normalizer.decode_action(predictions)
            self.queue.extend(torch.from_numpy(np.array(a,copy=True)) for a in actions[:self.action_steps])
        return self.queue.popleft()


def load_aligned_policy(checkpoint,device,action_steps):
    policy=SingleArmPolicy(checkpoint,device,action_steps)
    return policy,lambda value:value,lambda value:value,('overview',)
