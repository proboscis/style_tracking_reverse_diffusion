from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, List, Dict

import torch
from torch import Tensor
from torch.types import Device

from strd.adain import AdainStyleTransfer
from ldm.models.diffusion.ddpm import LatentDiffusion
from strd.input_preparation import DiffusionWrapperInputPreparer


class EpsModel(ABC):
    @abstractmethod
    def get_eps(self, x_t, t) -> Tensor:
        pass


@dataclass
class DefaultEpsModelForDiffusionWrapper(EpsModel):
    model: LatentDiffusion
    device: Device
    diffusion_wrapper_input_preparer: DiffusionWrapperInputPreparer
    unconditional_guidance_scale: float

    def _adain_apply_model(self, content_z, ddpm_t: int):
        uc_scale = self.unconditional_guidance_scale
        content_in = self.diffusion_wrapper_input_preparer.get_input_from_z(content_z, ddpm_t)
        e_t_uncond, e_t = self.model.model(*content_in.args, **content_in.kwargs).chunk(2)
        e_t = e_t_uncond + uc_scale * (e_t - e_t_uncond)
        assert isinstance(e_t, Tensor)
        return e_t

    def get_et(self, content_x, ddpm_t):
        return self._adain_apply_model(content_x, ddpm_t)

    def get_eps(self, x_t, ddpm_t) -> torch.Tensor:
        return self.get_et(x_t, ddpm_t)


@dataclass
class AdainEpsModelForDiffusionWrapper(EpsModel):
    adain_style_transfer: AdainStyleTransfer  # I guess this module should be a factory instead of an instance?
    unconditional_guidance_scale: float  # TODO move this caling inside the model...
    device: Any
    t_to_style_z: Callable
    diffusion_wrapper_input_preparer: DiffusionWrapperInputPreparer

    def get_eps(self, content_z, ddpm_t) -> torch.Tensor:
        uc_scale = self.unconditional_guidance_scale
        style_in = self.diffusion_wrapper_input_preparer.get_input_from_z(self.t_to_style_z(ddpm_t), ddpm_t)
        content_in = self.diffusion_wrapper_input_preparer.get_input_from_z(content_z, ddpm_t)
        e_t_uncond, e_t = self.adain_style_transfer.run(style_in, content_in).chunk(2)
        e_t = e_t_uncond + uc_scale * (e_t - e_t_uncond)
        assert isinstance(e_t, Tensor)
        return e_t



@dataclass
class CombinedEpsModel(EpsModel):
    """combines multiple eps models with weights"""
    models: List[EpsModel]
    weights: List[float]

    def get_eps(self, x_t, t) -> Tensor:
        e_ts = [m.get_eps(x_t, t) * self.weights[i] for i, m in enumerate(self.models)]
        return sum(e_ts)


@dataclass
class ContentModel(EpsModel):
    """
    calc eps from content_z_history at ddpm_step_t
    """
    model: LatentDiffusion
    t_to_content_z: Callable
    diffusion_wrapper_input_preparer: DiffusionWrapperInputPreparer

    def get_eps(self, x_t, ddpm_t) -> Tensor:
        content_hist_in = self.diffusion_wrapper_input_preparer.get_input_from_z(self.t_to_content_z(ddpm_t), ddpm_t)
        e_t = self.model.model(*content_hist_in.args, **content_hist_in.kwargs)
        assert isinstance(e_t, Tensor)
        return e_t


@dataclass
class ContentHistoryTracer(EpsModel):
    """
    return e_t from content_history
    """
    content_z_history: List[Tensor]
    content_z: Tensor
    ddpm_step_to_ddim_index: Dict[int, int]

    def get_eps(self, x_t, ddpm_t) -> Tensor:
        idx = self.ddpm_step_to_ddim_index[ddpm_t]
        current_content_z = self.content_z_history[idx]
        if idx == 0:
            restored_content_z = self.content_z
        else:
            restored_content_z = self.content_z_history[idx - 1]
        return restored_content_z - current_content_z
