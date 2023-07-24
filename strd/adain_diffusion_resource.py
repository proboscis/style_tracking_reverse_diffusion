from pinject_design import Design, Injected
from ray_proxy import Var, IRemoteInterpreter
from strd.adain_diffusion import SimplifiedImageToImage, LatentSpaceDecoder
from strd.adain_diffusion_design import img2img_adain_generation_design, encode_image

additional_design_for_adain_diffusion = Design(
    classes=[SimplifiedImageToImage, LatentSpaceDecoder]
).bind_instance(
    img2img_generation_design=img2img_adain_generation_design,
).bind_provider(
    latent_diffusion=lambda model: model,
    encode_image=encode_image,
).bind_class(
    latent_space_decoder=LatentSpaceDecoder,
    simplified_image_to_image=SimplifiedImageToImage,
)


def create_adain_diffusion(img2img_env: Var[IRemoteInterpreter]):
    env = img2img_env
    design = env["img2img_vars"]["design"] + additional_design_for_adain_diffusion
    ddim_sampler = design.provide(SimplifiedImageToImage)
    return ddim_sampler


def create_img2img_func(adain_img2img: Var[SimplifiedImageToImage]):
    return adain_img2img.run_img2img


def create_decode_func(img2img_env: Var[IRemoteInterpreter]):
    r_d = img2img_env["img2img_vars"]["design"] + additional_design_for_adain_diffusion
    r_g = r_d.to_graph()
    return r_g[LatentSpaceDecoder]


adain_img2img = Injected.bind(create_adain_diffusion)
img2img_decode = Injected.bind(create_decode_func)
img2img_run = Injected.bind(create_img2img_func)
