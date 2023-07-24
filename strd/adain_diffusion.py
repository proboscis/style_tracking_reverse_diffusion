from abc import ABC, abstractmethod
from contextlib import contextmanager, AbstractContextManager
from dataclasses import dataclass
from typing import Callable, Generator, List

import torch
from torch import Tensor
from torch.types import Device
from tqdm import tqdm

from strd.utilities.auto_image import AutoImage
from strd.utilities.rulebook import auto
from strd.adain import AdainStat
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.diffusionmodules.util import noise_like
from pinject_design import Design, Designed, Injected, EmptyDesign, injected_function
from pinject_design.di.graph import SessionValue, IObjectGraph
from pinject_design.di.proxiable import DelegatedVar
from pinject_design.di.util import instances
from strd.eps_models import EpsModel


@dataclass
class RestorationScope:
    latent_diffusion: LatentDiffusion
    precision_scope: AbstractContextManager  # should be "autocast"

    @contextmanager
    def work_scope(self):
        with torch.no_grad():
            with self.precision_scope("cuda"):
                with self.latent_diffusion.ema_scope():
                    yield


# we need to implement to variant of Adain.
# 1. adain on decoder.
#    - I feel this is not gonna work well, since gatys style transfer in latent space doesnt work well. we can finetune though.
# 2. adain on diffusion process.remote_
#    - This should work as expected, but it is not guaranteed.
# How can I implment them?
# we need two path. one for style and one for content.
# and we need to know at which layer we need the adain.
# so first get two models, accept target layers, and the replace the implementation of that layer with
# adain enabled one.


@dataclass
class SimpleDDIMSampler:
    sampler: DDIMSampler
    device: Device

    def p_sample_ddim(self,
                      x_t,
                      e_t,  # because we customize e_t a lot, e_t requires a lot of parameters.
                      index,
                      ):
        """
        :param index: index of steps in DDIM allocated steps
        :return:
        """
        assert isinstance(e_t, Tensor), type(e_t)
        assert isinstance(x_t, Tensor), type(x_t)
        b = e_t.shape[0]
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), self.sampler.ddim_alphas[index], device=self.device)
        a_prev = torch.full((b, 1, 1, 1), self.sampler.ddim_alphas_prev[index], device=self.device)
        sigma_t = torch.full((b, 1, 1, 1), self.sampler.ddim_sigmas[index], device=self.device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.sampler.ddim_sqrt_one_minus_alphas[index], device=self.device)
        # current prediction for x_0
        pred_x0 = (x_t - sqrt_one_minus_at * e_t) / a_t.sqrt()
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(x_t.shape, self.device)  # sigma_t is zero. since this is DDIM.
        # TODO how can we guide ddim? since std is 0?
        # => we can just add the gradient.
        # => so, we can actually introduce a sampler with guidance. options:
        # 1. introduce a guidance as a member for this sampler
        # 2. create a new sampler dedicated for it.
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0


class IGradientGuidance(ABC):
    @abstractmethod
    def provide_gradient(self, x0_hat: Tensor, index: int) -> Tensor:
        pass


@dataclass
class ScoreGuidance(IGradientGuidance):
    score_function: Callable[[Tensor, int], Tensor]

    def provide_gradient(self, x0_hat: Tensor, index) -> Tensor:
        score = self.score_function(x0_hat, index)
        gradient = torch.autograd.grad(score.mean(), x0_hat)[0]
        return gradient


@dataclass
class GuidedDDIMSampler:
    sampler: DDIMSampler
    device: Device
    gradient_guidance: IGradientGuidance

    def p_sample_ddim(self,
                      x_t,
                      e_t,  # because we customize e_t a lot, e_t requires a lot of parameters.
                      index,
                      ):
        """
        :param index: index of steps in DDIM allocated steps
        :return:
        """
        assert isinstance(e_t, Tensor), type(e_t)
        assert isinstance(x_t, Tensor)
        b = e_t.shape[0]
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), self.sampler.ddim_alphas[index], device=self.device)
        a_prev = torch.full((b, 1, 1, 1), self.sampler.ddim_alphas_prev[index], device=self.device)
        sigma_t = torch.full((b, 1, 1, 1), self.sampler.ddim_sigmas[index], device=self.device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.sampler.ddim_sqrt_one_minus_alphas[index], device=self.device)
        # current prediction for x_0
        with torch.enable_grad():
            e_t = e_t.detach()
            pred_x0 = (x_t - sqrt_one_minus_at * e_t) / a_t.sqrt()
            pred_x0 = pred_x0.detach().requires_grad_(True)
            # guidance calculation
            gradient = self.gradient_guidance.provide_gradient(pred_x0, index)
            pred_x0 = pred_x0.detach()
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t + gradient
        # currently the gradient has very little effect.
        # we need to give some scale.
        noise = sigma_t * noise_like(x_t.shape, self.device)  # sigma_t is zero. since this is DDIM.
        # now lets add the guidance.
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0


class IDDIMSampler(ABC):
    @abstractmethod
    def p_sample_ddim(self, x_t, t, ddim_index):
        pass


@dataclass
class DDIMSamplerWithModel(IDDIMSampler):
    # this is meant to be instantiated per diffusion session.
    device: Device
    simple_ddim_sampler: SimpleDDIMSampler
    eps_model: EpsModel

    def p_sample_ddim(self, x_t, ddpm_t, ddim_index):
        e_t = self.eps_model.get_eps(x_t, ddpm_t)
        assert isinstance(e_t, Tensor), f"e_t must be a Tensor. got:{type(e_t)}"
        x_prev, x_0_hat = self.simple_ddim_sampler.p_sample_ddim(x_t, e_t, ddim_index)
        return x_prev, x_0_hat


@dataclass
class DDIMSamplerEpsAdaIN(IDDIMSampler):
    """
    performs adain between epsilon domain
    the style comes from style_z_history. with ddim index. beware!
    """
    device: Device
    simple_ddim_sampler: SimpleDDIMSampler
    eps_model: EpsModel
    style_z_history: List[Tensor]

    def p_sample_ddim(self, x_t, ddpm_t, ddim_index):
        e_t = self.eps_model.get_eps(x_t, ddpm_t)
        e_t_for_style = self.eps_model.get_eps(self.style_z_history[ddim_index], ddpm_t)
        assert isinstance(e_t, Tensor), f"e_t must be a Tensor. got:{type(e_t)}"
        assert isinstance(e_t_for_style, Tensor), f"e_t_for_style must be a Tensor. got:{type(e_t_for_style)}"
        adain_stat = AdainStat.from_conv2d_out(e_t_for_style)
        new_e_t = adain_stat.transfer(e_t)

        x_prev, x_0_hat = self.simple_ddim_sampler.p_sample_ddim(x_t, new_e_t, ddim_index)
        return x_prev, x_0_hat


@dataclass
class DDIMSamplerZtAdaIN(IDDIMSampler):
    """
     performs adain between zt domain
     the style comes from style_z_history. with ddim index. beware!
     """
    device: Device
    simple_ddim_sampler: SimpleDDIMSampler
    eps_model: EpsModel
    style_z_history: List[Tensor]

    def p_sample_ddim(self, z_t, ddpm_t, ddim_index):
        e_t = self.eps_model.get_eps(z_t, ddpm_t)
        z_prev, x_0_hat = self.simple_ddim_sampler.p_sample_ddim(z_t, e_t, ddim_index)
        style_z_t = self.style_z_history[ddim_index]
        style_e_t = self.eps_model.get_eps(style_z_t, ddpm_t)
        style_z_prev, style_z0_hat = self.simple_ddim_sampler.p_sample_ddim(style_z_t, style_e_t, ddim_index)
        adain_stat = AdainStat.from_conv2d_out(style_z_prev)
        new_z_prev = adain_stat.transfer(z_prev)
        return new_z_prev, x_0_hat  # beware x_0_hat has no meaning here.


@dataclass
class DDIMSamplerInitialZtAdaIN(IDDIMSampler):
    """
  performs adain between zt domain
  the style comes from style_z_history. with ddim index. beware!
  """
    device: Device
    simple_ddim_sampler: SimpleDDIMSampler
    eps_model: EpsModel
    style_z_history: List[Tensor]
    ddim_t_start: int

    def p_sample_ddim(self, z_t, ddpm_t, ddim_index):
        e_t = self.eps_model.get_eps(z_t, ddpm_t)
        z_prev, x_0_hat = self.simple_ddim_sampler.p_sample_ddim(z_t, e_t, ddim_index)
        if ddim_index == self.ddim_t_start:
            style_z_t = self.style_z_history[ddim_index]
            style_e_t = self.eps_model.get_eps(style_z_t, ddpm_t)
            style_z_prev, style_z0_hat = self.simple_ddim_sampler.p_sample_ddim(style_z_t, style_e_t, ddim_index)
            adain_stat = AdainStat.from_conv2d_out(style_z_prev)
            new_z_prev = adain_stat.transfer(z_prev)
            return new_z_prev, x_0_hat  # beware x_0_hat has no meaning here.
        else:
            return z_prev, x_0_hat


@dataclass
class SimplifiedDdimSampler:
    # I think this is getting printed every time.
    ddim_sampler: IDDIMSampler
    encode_image: Callable[[AutoImage], Tensor]
    ddim_step_pairs: list
    restoration_scope: RestorationScope
    sampler: DDIMSampler
    ddim_t_start: int
    batch_size: int
    device: Device

    def generate(self, img: AutoImage):
        for item in self.run_generator(img):
            pass
        return item

    def run_generator(self, img: AutoImage):
        with self.restoration_scope.work_scope():
            z0 = self.encode_image(img)
            t_tensor = torch.tensor([self.ddim_t_start] * self.batch_size).to(self.device)
            z_t = self.sampler.stochastic_encode(z0, t_tensor)

            for ddpm_t, ddim_index in tqdm(self.ddim_step_pairs, desc="ddim sampling"):
                z_t, z_0_hat = self.ddim_sampler.p_sample_ddim(z_t, ddpm_t, ddim_index)
                yield dict(
                    ddpm_t=ddpm_t,
                    ddim_index=ddim_index,
                    z_t=z_t,
                    z_0_hat=z_0_hat,
                    z_0=z0
                )

    def __repr__(self):
        return f"""SimplifiedDdimSampler(contents ommitted)"""



@dataclass
class SimplifiedImageToImage:
    session: IObjectGraph
    img2img_generation_design: Design

    def __post_init__(self):
        self.designed_sampler: Designed = Designed \
            .bind(Injected.bind(SimplifiedDdimSampler)) \
            .override(self.img2img_generation_design)

    def run_generator(self, img: AutoImage, prompt, strength, overrides: Design = None) -> Generator[
        DelegatedVar[SessionValue[dict]], None, None]:
        yield from self.sampling_session(prompt, strength, overrides).run_generator(img)

    def run_img2img(self, img: AutoImage, prompt, strength, overrides=None) -> DelegatedVar[SessionValue[dict]]:
        for item in self.run_generator(img, prompt, strength, overrides):
            pass
        return item

    def run_img2img_direct(self, img, prompt, strength, overrides=None):
        return self.run_img2img(img, prompt, strength, overrides).eval()

    def sampling_session(self, prompt, strength, overrides: Design = None) -> DelegatedVar[SessionValue[Callable]]:
        overrides = overrides or Design()
        applied: Designed = self.designed_sampler.override(
            EmptyDesign.bind_instance(
                prompt=prompt,
                strength=strength
            ) + overrides
        )
        if 'style_img' in overrides:
            assert overrides.provide("style_img") == self.session.child_session(overrides)[
                'style_img'], "style img in child session is not the same, some thing weird is happening!"

        return self.session.sessioned(applied)

    def run_img2img_direct_generator(self, img, prompt, strength, overrides=None):
        child_overrides = self.img2img_generation_design + instances(
            prompt=prompt,
            strength=strength
        ) + overrides
        sess = self.session.child_session(child_overrides)
        sds: SimplifiedDdimSampler = sess[SimplifiedDdimSampler]
        for item in sds.run_generator(img):
            assert isinstance(item, dict), f"item must be a dict. got:{type(item)}"
            yield item

    def run_img2img_direct2(self, img, prompt, strength, overrides=None):
        for item in self.run_img2img_direct_generator(img, prompt, strength, overrides):
            pass
        return item


@dataclass
class LatentSpaceDecoder:
    latent_diffusion: LatentDiffusion

    def decode(self, z: Tensor, with_grad=False):
        if with_grad:
            z = 1.0 / self.latent_diffusion.scale_factor * z
            x = self.latent_diffusion.first_stage_model.decode(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        else:
            x = self.latent_diffusion.decode_first_stage(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        return x

    def decode_auto(self, z: Tensor) -> AutoImage:
        return AutoImage(auto("torch,float32,BCHW,RGB,0_1", self.decode(z)))


