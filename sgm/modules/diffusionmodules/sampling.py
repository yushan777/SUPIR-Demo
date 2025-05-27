"""
    Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""
from typing import Dict, Union
import torch
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

from ...modules.diffusionmodules.sampling_utils import (
    # get_ancestral_step,
    # linear_multistep_coeff,
    to_d,
    # to_neg_log_sigma,
    # to_sigma,
)

from ...util import append_dims, default, instantiate_from_config
# from k_diffusion.sampling import get_sigmas_karras, BrownianTreeNoiseSampler

#  =================================================================

#  =================================================================


DEFAULT_GUIDER = {"target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"}


class BaseDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        verbose: bool = True,
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(
            default(
                guider_config,
                DEFAULT_GUIDER,
            )
        )
        self.verbose = verbose
        self.device = device

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        uc = default(uc, cond)

        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, denoiser, sigma, cond, uc):
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc))
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas-1,  # Changed from num_sigmas so progress shows same total as --edm_steps
                desc=f"Sampling with {self.__class__.__name__}: ",
                bar_format='{desc}: |{bar}| {percentage:3.0f}% • Step {n_fmt}/{total_fmt} • {elapsed}<{remaining}',
                colour='green',
                smoothing=1.0,                
            )
        return sigma_generator


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d

class RestoreEDMSampler(SingleStepDiffusionSampler):
    def __init__(
        self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, restore_cfg=4.0,
            restore_cfg_s_tmin=0.05, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.restore_cfg = restore_cfg
        self.restore_cfg_s_tmin = restore_cfg_s_tmin
        self.sigma_max = 14.6146

    def denoise(self, x, denoiser, sigma, cond, uc, control_scale=1.0):
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), control_scale)
        denoised = self.guider(denoised, sigma)
        return denoised

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0, x_center=None, eps_noise=None,
                     control_scale=1.0, control_scale_start=0.0):
        
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            if eps_noise is not None:
                eps = eps_noise * self.s_noise
            else:
                eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        # if control_scale_start - control_scale are same then it's same as having linear control scale turned off
        control_scale = (sigma[0].item() / self.sigma_max) * (control_scale_start - control_scale) + control_scale

        denoised = self.denoise(x, denoiser, sigma_hat, cond, uc, control_scale=control_scale)

        if (next_sigma[0] > self.restore_cfg_s_tmin) and (self.restore_cfg > 0):
            d_center = (denoised - x_center)
            denoised = denoised - d_center * ((sigma.view(-1, 1, 1, 1) / self.sigma_max) ** self.restore_cfg)

        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)
        x = self.euler_step(x, d, dt)
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, x_center=None, control_scale=1.0, control_scale_start=0.0):
        
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )


        for _idx, i in enumerate(self.get_sigma_gen(num_sigmas)):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
                x_center,
                control_scale=control_scale,
                control_scale_start=control_scale_start,
            )

                 
        return x


class TiledRestoreEDMSampler(RestoreEDMSampler):
    def __init__(self, tile_size=128, tile_stride=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.tile_weights = gaussian_weights(self.tile_size, self.tile_size, 1)

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, x_center=None, control_scale=1.0, control_scale_start=0.0):
        
        use_local_prompt = isinstance(cond, list)
        b, _, h, w = x.shape
        latent_tiles_iterator = _sliding_windows(h, w, self.tile_size, self.tile_stride)
        tile_weights = self.tile_weights.repeat(b, 1, 1, 1)
        if not use_local_prompt:
            LQ_latent = cond['control']
        else:
            assert len(cond) == len(latent_tiles_iterator), "Number of local prompts should be equal to number of tiles"
            LQ_latent = cond[0]['control']
        clean_LQ_latent = x_center
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

     

        for _idx, i in enumerate(self.get_sigma_gen(num_sigmas)):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            x_next = torch.zeros_like(x)
            count = torch.zeros_like(x)
            eps_noise = torch.randn_like(x)
            for j, (hi, hi_end, wi, wi_end) in enumerate(latent_tiles_iterator):
                x_tile = x[:, :, hi:hi_end, wi:wi_end]
                _eps_noise = eps_noise[:, :, hi:hi_end, wi:wi_end]
                x_center_tile = clean_LQ_latent[:, :, hi:hi_end, wi:wi_end]
                if use_local_prompt:
                    _cond = cond[j]
                else:
                    _cond = cond
                _cond['control'] = LQ_latent[:, :, hi:hi_end, wi:wi_end]
                uc['control'] = LQ_latent[:, :, hi:hi_end, wi:wi_end]
                _x = self.sampler_step(
                    s_in * sigmas[i],
                    s_in * sigmas[i + 1],
                    denoiser,
                    x_tile,
                    _cond,
                    uc,
                    gamma,
                    x_center_tile,
                    eps_noise=_eps_noise,
                    control_scale=control_scale,
                    control_scale_start=control_scale_start,
                )
                x_next[:, :, hi:hi_end, wi:wi_end] += _x * tile_weights
                count[:, :, hi:hi_end, wi:wi_end] += tile_weights
            x_next /= count
            x = x_next

        return x

def gaussian_weights(tile_width, tile_height, nbatches):
    """Generates a gaussian mask of weights for tile contributions"""
    from numpy import pi, exp, sqrt
    import numpy as np

    latent_width = tile_width
    latent_height = tile_height

    var = 0.01
    midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
    x_probs = [exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)) / sqrt(2 * pi * var)
               for x in range(latent_width)]
    midpoint = latent_height / 2
    y_probs = [exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var)) / sqrt(2 * pi * var)
               for y in range(latent_height)]

    weights = np.outer(y_probs, x_probs)
    return torch.tile(torch.tensor(weights, device='cuda'), (nbatches, 4, 1, 1))


def _sliding_windows(h: int, w: int, tile_size: int, tile_stride: int):
    hi_list = list(range(0, h - tile_size + 1, tile_stride))
    if (h - tile_size) % tile_stride != 0:
        hi_list.append(h - tile_size)

    wi_list = list(range(0, w - tile_size + 1, tile_stride))
    if (w - tile_size) % tile_stride != 0:
        wi_list.append(w - tile_size)

    coords = []
    for hi in hi_list:
        for wi in wi_list:
            coords.append((hi, hi + tile_size, wi, wi + tile_size))
    return coords
