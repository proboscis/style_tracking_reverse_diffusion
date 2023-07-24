from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from strd.adain import Inputs
from ldm.models.diffusion.ddpm import LatentDiffusion


@dataclass
class DiffusionWrapperInputPreparer:
    model: LatentDiffusion
    conditional_condition: Any
    unconditional_condition: Any
    device: Any

    def get_condition(self, cond):
        match self.model.model.conditioning_key:
            # I think we should abstract this away...by wrapping the DiffusionWrapper
            case "concat":
                cond = dict(c_concat=cond)
            case _:
                if not isinstance(cond, list):
                    cond = [cond]
                cond = dict(c_crossattn=cond)
        return cond

    def get_input(self, x: Tensor, c: Tensor, uc: Tensor, t: Tensor):
        assert isinstance(x, Tensor)
        assert isinstance(c, Tensor)
        assert isinstance(uc, Tensor)
        assert isinstance(t, Tensor)
        b, *_, device = *x.shape, x.device
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        c_in = torch.cat([uc, c])
        c_in = self.get_condition(c_in)
        return Inputs(x_in, t_in, **c_in)

    def get_input_from_z(self, z: Tensor, ddpm_t):
        c = self.conditional_condition
        uc = self.unconditional_condition
        t_tensor = torch.full((z.shape[0],), ddpm_t, device=self.device, dtype=torch.long)
        return self.get_input(z, c, uc, t_tensor)


