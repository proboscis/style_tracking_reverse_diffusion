from dataclasses import dataclass

from pinject_design import Design
from strd.utilities.auto_image import AutoImage


@dataclass
class Img2ImgInput:
    style: AutoImage
    content: AutoImage
    prompt: str
    strength: float
    overrides: Design = Design()
