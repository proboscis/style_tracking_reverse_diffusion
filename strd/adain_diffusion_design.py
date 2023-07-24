from typing import List, Callable

import PIL
import numpy as np
import torch
from PIL import Image
#from PIL.Image import Resampling
from einops import repeat
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusion
from omni_converter import IAutoData
from pinject_design import injected_function, Design, Injected
from pinject_design.di.proxiable import DelegatedVar

from strd.adain import AdainStyleTransfer
from strd.adain_diffusion import SimpleDDIMSampler, DDIMSamplerWithModel, LatentSpaceDecoder, SimplifiedDdimSampler, \
    RestorationScope
from strd.conv_util import list_convs
from strd.ddim_encoding import DdimEncodingHistoryProvider, TensorIndexExtractor
from strd.eps_models import AdainEpsModelForDiffusionWrapper
from strd.input_preparation import DiffusionWrapperInputPreparer
from strd.utilities.auto_image import AutoImage


def prep_image(path_or_img):
    if isinstance(path_or_img, str):
        img = Image.open(path_or_img)
    else:
        img = path_or_img

    image = img.convert("RGB")
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def provide_style_mapping(latent_diffusion: LatentDiffusion):
    conv_names = list_convs(latent_diffusion.model)
    return {conv: conv for conv in conv_names}


def provide_ddim_step_pairs(ddim_t_start, sampler: DDIMSampler, opt):
    sampler.make_schedule(
        ddim_num_steps=opt.ddim_steps,
        ddim_eta=opt.ddim_eta,
        verbose=False
    )
    assert isinstance(sampler, DDIMSampler)
    timesteps = sampler.ddim_timesteps
    timesteps = timesteps[:ddim_t_start]  # t_start in ddim space, timestep in ddpm space.
    # so, if we know t, then ...
    n_steps = timesteps.shape[0]
    mapping = []
    for i, step in enumerate(np.flip(timesteps)):
        index = n_steps - i - 1
        # well, this calculation is the same as just providing i, right?
        # mapping[step] = index
        mapping.append((step, index))
    return mapping


def provide_ddim_step_mapping(ddim_step_pairs: list):
    from loguru import logger
    logger.info("providing ddim_step_mapping")
    logger.info(f"ddim_step_mapping: {ddim_step_pairs}")
    return dict(ddim_step_pairs)


def provide_ddim_t_start(strength, opt):
    return min(max(int(strength * opt.ddim_steps), 0), opt.ddim_steps)


ddim_encoder = Injected.by_name("ddim_encoding_history_provider").proxy

content_z_encoding_history: DelegatedVar = ddim_encoder.get_encoding_history(Injected.by_name("content_z"),
                                                                             Injected.by_name("ddim_t_start"))
style_z_encoding_history = ddim_encoder.get_encoding_history(Injected.by_name("style_z"),
                                                             Injected.by_name("ddim_t_start"))
# here we are using 'by_name' to explicitly cache the variable in design.
# so that the encoding_history will be consistent throughout the session.
content_z_history = Injected.by_name("content_z_encoding_history").map(lambda hist: [s.z_t for s in hist])
style_z_history = Injected.by_name("style_z_encoding_history").map(lambda hist: [s.z_t for s in hist])


def get_ddpm_t_to_z(ddpm_step_to_ddim_index, z_history: List):
    def impl(t):
        index = ddpm_step_to_ddim_index[t]
        return z_history[index]

    return impl


t_to_style_z: Injected[Callable] = Injected.bind(get_ddpm_t_to_z, z_history="style_z_history")
t_to_content_z: Injected[Callable] = Injected.bind(get_ddpm_t_to_z, z_history="content_z_history")


@injected_function
def encode_image(_model: LatentDiffusion, _device, _batch_size, image: AutoImage):
    assert isinstance(image, IAutoData), f"image should be IAutoData, but got {type(image)}"
    if not isinstance(image, AutoImage):
        image = AutoImage(image)
    image.assert_one_image()
    img = image.first("image,RGB,RGB")
    init_image = prep_image(img).to(_device)
    init_image = repeat(init_image, '1 ... -> b ...', b=_batch_size)
    res = _model.get_first_stage_encoding(_model.encode_first_stage(init_image))
    return res


def provide_style_z(encode_image, style_img: AutoImage):
    return encode_image(style_img)


def provide_content_z(encode_image, content_img: AutoImage):
    # well I forgot what this is but I can use it. :)
    return encode_image(content_img)


img2img_adain_generation_design = Design(
    classes=[
        SimpleDDIMSampler,
        AdainEpsModelForDiffusionWrapper,
        DDIMSamplerWithModel,
        DdimEncodingHistoryProvider,
        LatentSpaceDecoder,
        DiffusionWrapperInputPreparer,
        TensorIndexExtractor
    ]
).bind_class(
    eps_model=AdainEpsModelForDiffusionWrapper,
    # eps_model=DefaultEpsModelForDiffusionWrapper,
    adain_style_transfer=AdainStyleTransfer,
    simplified_ddim_sampler=SimplifiedDdimSampler,
    simple_ddim_sampler=SimpleDDIMSampler,
    ddim_sampler=DDIMSamplerWithModel,
    restoration_scope=RestorationScope
).bind_instance(
    batch_size=1,
    negative_prompt=""
).bind_provider(
    adain_style_model=lambda latent_diffusion: latent_diffusion.model,
    adain_target_model=lambda latent_diffusion: latent_diffusion.model,
    unconditional_guidance_scale=lambda opt: opt.scale,
    conditional_condition=lambda model, prompt: model.get_learned_conditioning([prompt]),
    unconditional_condition=lambda model, negative_prompt: model.get_learned_conditioning([negative_prompt]),
    t_to_style_z=t_to_style_z,
    t_to_content_z=t_to_content_z,
    ddim_step_pairs=provide_ddim_step_pairs,
    ddpm_step_to_ddim_index=provide_ddim_step_mapping,
    ddim_t_start=provide_ddim_t_start,
    ddpm_t_start=lambda ddim_step_pairs: ddim_step_pairs[0][0],
    style_z=provide_style_z,
    style_z_encoding_history=style_z_encoding_history,
    style_z_history=style_z_history,
    content_z=provide_content_z,
    content_z_encoding_history=content_z_encoding_history,
    content_z_history=content_z_history,
    latent_diffusion=lambda model: model,
    encode_image=encode_image
)


def provide_background_z(encode_image, base_image):
    return encode_image(AutoImage(base_image))


def provide_ddpm_t_to_background_zt(
        base_z,
        ddim_encoding_history_provider,
        ddim_t_start,
        ddpm_step_to_ddim_index
):
    history = ddim_encoding_history_provider.get_z_history(base_z, ddim_t_start)

    def impl(ddpm_t):
        return history[ddpm_step_to_ddim_index[ddpm_t]]

    return impl
