"""Dependency-free cosine DDPM forward process and deterministic DDIM updates.

Equations checked against the official diffusers DDPM/DDIM schedulers:
https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py
https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py
This local implementation uses epsilon prediction, eta=0, and clips predicted
x0 to [-1, 1] by default, matching the dataset's train-only min/max scaling.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class CosineDiffusionSchedule(nn.Module):
    """Squared-cosine beta schedule with max_beta=0.999 and alpha_bar(-1)=1."""

    def __init__(self, diffusion_steps: int, clip_sample: bool = True) -> None:
        super().__init__()
        if isinstance(diffusion_steps, bool) or not isinstance(diffusion_steps, int) or diffusion_steps < 1:
            raise ValueError("diffusion_steps must be a positive integer")
        self.diffusion_steps = diffusion_steps
        if not isinstance(clip_sample, bool):
            raise ValueError("clip_sample must be boolean")
        self.clip_sample = clip_sample

        def alpha_bar(fraction: float) -> float:
            return math.cos((fraction + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = torch.tensor([
            min(1 - alpha_bar((i + 1) / diffusion_steps) / alpha_bar(i / diffusion_steps), 0.999)
            for i in range(diffusion_steps)
        ], dtype=torch.float32)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", (1 - betas).cumprod(0))

    def add_noise(self, clean: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """q(x_t|x_0) = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*epsilon."""
        alpha = self.alphas_cumprod[timesteps].to(dtype=clean.dtype)
        alpha = alpha.reshape(-1, *([1] * (clean.ndim - 1)))
        return alpha.sqrt() * clean + (1 - alpha).sqrt() * noise

    def inference_timesteps(self, num_inference_steps: int) -> tuple[int, ...]:
        """A trailing grid starting at T-1, including a one-step T-1 -> clean case.

        Each DDIM update uses the actual next entry of this grid. This avoids a
        floor-stride mismatch when inference_steps does not divide T exactly.
        """
        if (isinstance(num_inference_steps, bool) or not isinstance(num_inference_steps, int)
                or not 1 <= num_inference_steps <= self.diffusion_steps):
            raise ValueError("num_inference_steps must be in [1, diffusion_steps]")
        stride = self.diffusion_steps / num_inference_steps
        return tuple(round(self.diffusion_steps - i * stride) - 1 for i in range(num_inference_steps))

    def ddim_step(self, epsilon: Tensor, timestep: int, previous_timestep: int,
                  sample: Tensor) -> Tensor:
        """DDIM (eta=0): reconstruct x0, then move to the previous alpha_bar.

        No fresh noise is drawn here, so the initial sample fully determines
        inference randomness. The final step uses alpha_bar=1 exactly.
        """
        alpha = self.alphas_cumprod[timestep].to(dtype=sample.dtype)
        previous_alpha = (self.alphas_cumprod[previous_timestep].to(dtype=sample.dtype)
                          if previous_timestep >= 0 else sample.new_tensor(1.0))
        epsilon = epsilon.to(dtype=sample.dtype)
        clean = (sample - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()
        if self.clip_sample:
            clean = clean.clamp(-1, 1)
        return previous_alpha.sqrt() * clean + (1 - previous_alpha).sqrt() * epsilon
