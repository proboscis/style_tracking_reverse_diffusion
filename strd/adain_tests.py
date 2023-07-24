from pinject_design import Injected
from strd.utilities.auto_image import AutoImage
from strd.adain_options import adain_options
from strd.distributed_adain import adain_img2img_decoded

dummy = AutoImage.auto("color_code","#ffffff")
converted:Injected = adain_img2img_decoded(
    style=dummy,
    content=dummy,
    prompt="",
    strength=0.5,
    style_mapping=adain_options.proxy['all'].eval().map(lambda layers:{l:l for l in layers}),
)

