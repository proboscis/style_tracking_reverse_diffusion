from lightning_fabric import seed_everything

from strd.util import ensure_image, resolve_design, transfer_style
from pinject_design import Design, injected_function, Injected
from pinject_design.di.injected import injected
from pinject_design.di.util import instances, classes, providers
from pinject_design.run_config_utils import run_main
from strd.adain_diffusion import LatentSpaceDecoder, SimplifiedImageToImage, SimplifiedDdimSampler
from strd.adain_diffusion_design import img2img_adain_generation_design
from strd.adain_diffusion_resource import additional_design_for_adain_diffusion
from strd.designs import default_design
from strd.stable_diffusion_adapter import serve_img2img


@injected_function
def local_adain_img2img_decoded(
        random_seed: int,
        latent_space_decoder: LatentSpaceDecoder,
        simplified_ddim_sampler: SimplifiedDdimSampler,
        /,
        content,
):
    seed_everything(random_seed)
    for item in simplified_ddim_sampler.run_generator(content):
        pass
    img = latent_space_decoder.decode_auto(item["z_t"])
    return img.auto_first("image,RGB,RGB")


single_gpu_design = default_design + instances(
    gpu='dummy',
    random_seed=42,
) + providers(
    resolve_design=resolve_design,
    adain_img2img_decoded=local_adain_img2img_decoded,
    img2img_env=Injected.bind(
        serve_img2img,
        working_dir=Injected.by_name("img2img_sd_working_dir"),
        argv=Injected.by_name("img2img_sd_argv"),
    ),
    model=lambda img2img_env: img2img_env['model'],
    precision_scope=injected("img2img_env")['precision_scope'],
    device=injected("img2img_env")['device'],
    opt=injected("img2img_env")['opt'],
    sampler = injected("img2img_env")['sampler'],
) + classes(
) + img2img_adain_generation_design + additional_design_for_adain_diffusion

__default_design_paths__ = ['examples.single_gpu_main.single_gpu_design']

example_32: Injected = local_adain_img2img_decoded(
    content=Injected.bind(lambda :ensure_image("./example_data/image_pairs/content_2.png")),
    style_img = Injected.bind(lambda :ensure_image("./example_data/image_pairs/style_2.png")),
    prompt="",
    strength=0.5,
    style_mapping=dict(),# must be the names of convolutional layers
).show_plot()
# TODO make run_main accept kwargs for design overrides
if __name__ == '__main__':
    run_main()
