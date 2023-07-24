from pathlib import Path
from typing import Union

from PIL.Image import Image

from pinject_design import injected_function, Design
from pinject_design.di.util import instances

from strd.utilities.auto_image import AutoImage
from strd.utilities.rulebook import identify_image

ImageLike = Union[str, Image, AutoImage, Path]


def ensure_image(image: ImageLike) -> AutoImage:
    match image:
        case Path() as p:
            assert p.exists(), f"Path {p} does not exist, but treated as an image path."
            return AutoImage.auto("image_path", str(p)).convert('image,RGB,RGB')
        case str() as p:
            assert Path(p).exists(), f"Path {p} does not exist, but treated as an image path."
            return AutoImage.auto("image_path", p).convert('image,RGB,RGB')
        case _:
            return identify_image(image)


@injected_function
def resolve_design(session, /, overrides: Union[str, Design]) -> Design:
    from strd.designs import adain_img2img_overrides
    match overrides:
        case "all_layers":
            return session[adain_img2img_overrides]
        case "no_layers":
            return instances(
                style_mapping=dict()
            )


@injected_function
def transfer_style(
        adain_img2img_decoded,
        resolve_design,
        /,  # positional only arguments are injected.
        style: ImageLike,
        content: ImageLike,
        prompt="",
        strength=0.5,
        overrides: Union[str, Design] = "all_layers"
) -> AutoImage:
    """
    :param adain_img2img_decoded: a service which gets injected
    :param resolve_design: injected util function
    :param style: path or an image
    :param content: path or an image
    :param prompt: optional text to guide the diffusion process
    :param strength:
    :param overrides: "all_layers" or "no_layers", or a Design instance
    all_layers: use all layers for AdaIN
    no_layers: use no layers for AdaIN
    overrides: can be used to swap anything used inside 'adain_img2img_decoded' function.
    :return:
    """
    overrides = resolve_design(overrides)
    res: AutoImage = adain_img2img_decoded(
        style=ensure_image(style),
        content=ensure_image(content),
        prompt=prompt,
        strength=strength,
        style_mapping=dict(),
        overrides=overrides
    )
    return res
