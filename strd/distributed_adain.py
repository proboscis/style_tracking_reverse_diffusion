from lightning_fabric import seed_everything

from pinject_design import injected_function, Design, Injected
from ray_proxy.resource_scheduler_client import ResourceSchedulerClient
from strd.adain_diffusion import LatentSpaceDecoder, SimplifiedImageToImage
from strd.adain_diffusion_resource import additional_design_for_adain_diffusion
from strd.utilities.auto_image import AutoImage


@injected_function
def adain_img2img_decoded(_sch: ResourceSchedulerClient,
                          /,
                          style: AutoImage,
                          content: AutoImage,
                          prompt: str,
                          strength: float,
                          style_mapping: dict,
                          overrides: Design = None,
                          seed: int = 42
                          ) -> AutoImage:
    """
    a function to do adain_img2img style transfer on a cluster
    :param _sch:
    :param style:
    :param content:
    :param prompt:
    :param strength:
    :param style_mapping:
    :param overrides:
    :param seed:
    :return:
    """
    with _sch["img2img_session"] as sess:
        g = sess.child_session(additional_design_for_adain_diffusion)
        override = Design().bind_instance(
            style_img=style,
            style_mapping=style_mapping
        ) + (overrides if overrides is not None else Design())
        g.env.put(seed_everything)(seed)
        img = g[img2img_decoded(
            Injected.bind(LatentSpaceDecoder),
            Injected.bind(SimplifiedImageToImage),
            content, prompt, strength, override
        )]
        from loguru import logger
        return img.fetch()  # you must fetch before the session ends!


@injected_function
def img2img_decoded(
        decoder: LatentSpaceDecoder,
        sampler: SimplifiedImageToImage,
        content, prompt, strength, override
) -> "AutoImage":
    assert isinstance(sampler, SimplifiedImageToImage)
    data = sampler.run_img2img_direct2(
        content, prompt, strength, override
    )
    img = decoder.decode_auto(data["z_t"])
    return img.auto_first("image,RGB,RGB")
