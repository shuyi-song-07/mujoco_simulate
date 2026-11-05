"""Original diffusers DDPM scheduler configuration, without a local rewrite."""
from diffusers import DDPMScheduler


def make_noise_scheduler(diffusion_steps=100, clip_sample=True):
    return DDPMScheduler(num_train_timesteps=diffusion_steps, beta_start=1e-4, beta_end=0.02,
        beta_schedule="squaredcos_cap_v2", variance_type="fixed_small",
        clip_sample=clip_sample, prediction_type="epsilon")
