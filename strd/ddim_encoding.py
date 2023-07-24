from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor
from tqdm import tqdm

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules.util import extract_into_tensor


@dataclass
class DdimEncodingSample:
    ddim_index: int
    ddpm_t: int
    z_t: Tensor
    e_t: Tensor


@dataclass
class TensorIndexExtractor:
    device: str
    batch_size: int

    def extract(self, src, idx: int, shape):
        i = torch.tensor([idx] * self.batch_size).to(self.device)  # shape [B,1] tensor?
        return extract_into_tensor(src, i, shape)


class IDdimEncodingHistoryProvider(metaclass=ABCMeta):
    @abstractmethod
    def get_z_history(self, z0, t_start)->List[Tensor]:
        pass

    @abstractmethod
    def get_encoding_history(self, z0, t_start)->List[DdimEncodingSample]:
        pass

@dataclass
class DdimDirectEncodingHistoryProvider(IDdimEncodingHistoryProvider):
    """a class to provide a history of encoding by directly sampling it from one noise."""
    sampler: DDIMSampler
    device: str
    batch_size: int
    tensor_index_extractor: TensorIndexExtractor

    def get_z_history(self, z0, t_start) -> List[Tensor]:
        history = self.get_encoding_history(z0, t_start)
        return [sample.z_t for sample in history]

    def get_encoding_history(self, z0, t_start) -> List[DdimEncodingSample]:
        from loguru import logger
        timesteps = self.sampler.ddim_timesteps[:t_start]
        zt_history = []
        for i, t in enumerate(tqdm(timesteps, desc="directory encoding each step")):  # t is in ddpm step space
            t_enc = torch.tensor([i] * self.batch_size).to(self.device)
            logger.info(f"t_enc:{t_enc}")
            z_t = self.sampler.stochastic_encode(z0,t_enc)
            zt_history.append(DdimEncodingSample(
                ddim_index=i,
                ddpm_t=t,
                z_t=z_t,
                e_t=None# let's just leave it None for now
            ))
        return zt_history


@dataclass
class DdimEncodingHistoryProvider(IDdimEncodingHistoryProvider):
    sampler: DDIMSampler
    device: str
    batch_size: int
    tensor_index_extractor: TensorIndexExtractor

    def get_z_history(self, z0, t_start) -> List[Tensor]:
        history = self.get_encoding_history(z0, t_start)
        return [sample.z_t for sample in history]

    def get_encoding_history(self, z0, t_start) -> List[DdimEncodingSample]:
        timesteps = self.sampler.ddim_timesteps[:t_start]
        alphas_cumprod = self.sampler.ddim_alphas
        alphas_cumprod_prev = torch.tensor(self.sampler.ddim_alphas_prev).float().to(self.device)
        alpha_ratio = alphas_cumprod / alphas_cumprod_prev
        mean_amount = torch.sqrt(alpha_ratio)
        noise_amount = torch.sqrt(1 - alpha_ratio)
        zt_history = []
        zt = z0
        for i, t in enumerate(tqdm(timesteps, desc="encoding step by step")):  # t is in ddpm step space
            ma = self.tensor_index_extractor.extract(mean_amount, i, z0.shape)
            na = self.tensor_index_extractor.extract(noise_amount, i, z0.shape)
            print(t, ma, na)
            noise = torch.randn_like(z0)
            zt = ma * zt + na * noise
            # so this 'noise' is not what the model is trying to predict...
            # because this noise cannot reconstruct the z0.
            # the actual noises that were target of the model are the sum of noise history.
            # mathematically the sum of noise history is just a gaussian, but
            # the gaussians are mixed with multiple scales, like
            # alpha * et + alpha2*2t+1....so,,, My assumption is that the target noise from q_sample is time dependent??
            # TODO the actual e_t is (z_0 - z_t) times something(which makes the mean/std 0/1).
            zt_history.append(
                DdimEncodingSample(
                    ddim_index=i,
                    ddpm_t=t,
                    z_t=zt,
                    e_t=noise
                )
            )
        # return zt,noise_history
        # I think we need to keep the history of the noise itself for reconstruction
        return zt_history
