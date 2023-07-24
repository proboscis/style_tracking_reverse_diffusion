import hashlib
from typing import Union, List

import PIL
import ray

from attr import define, field
from attrs import validators
from ray import ObjectRef

from omni_converter import IAutoData, AutoRuleBook
from omni_converter.solver.astar import Converter
from omni_cv_rules.coconut.convert import ch_splitter
from omni_cv_rules.coconut.omni_converter import AutoList
from strd.utilities.rulebook import auto, identify_image
from ray_proxy import Var


@define
class AutoImage(IAutoData):
    src: IAutoData = field(validator=[validators.instance_of(IAutoData)])

    @property
    def value(self):
        return self.src.value

    @property
    def format(self):
        return self.src.format

    def to(self, format):
        return self.src.to(format)

    def convert(self, format) -> "AutoImage":
        return AutoImage(self.src.convert(format))

    def converter(self, format) -> Converter:
        return self.src.converter(format)

    def map(self, f, format=None) -> "AutoImage":
        return AutoImage(self.src.map(f, format))

    def override(self, rule: AutoRuleBook) -> "AutoImage":
        return AutoImage(self.src.override(rule))

    def cast(self, new_format) -> "AutoImage":
        return AutoImage(self.src.cast(new_format))

    def histogram(self, show=True):
        from matplotlib import pyplot as plt
        fig = plt.figure()
        aim = self.convert(type="numpy", arrange="BHWC")
        ary = aim.value
        chs = ch_splitter(aim.format["ch_rpr"])
        from loguru import logger
        logger.info(f"histogram src shape:{ary.shape}")
        logger.info(f"src channels:{','.join(chs)}")
        for i, ch in enumerate(chs):
            plt.hist(ary[:, :, :, i].flatten(), bins=100, label=ch)
        plt.legend()
        if show:
            plt.show()
        return fig

    def color_map(self, show=True, value_range="0_1", vmin=None, vmax=None):
        from matplotlib import pyplot as plt

        aim = self.convert(type="numpy", arrange="BHWC")
        # ary = aim.value
        chs = ch_splitter(aim.format["ch_rpr"])
        figs = []
        for i, c in enumerate(chs):
            figs.append(plt.figure())
            tgt = aim.to(f"numpy,float32,HW,{c},{value_range}")
            plt.imshow(tgt, vmin=vmin, vmax=vmax)
            # plt.imshow(aim.to((f"numpy,uint8,HW,{c},0_255")))
            plt.colorbar()
        if show:
            plt.show()
        return figs

    def as_spectrum_img(self):
        return self.convert("spectrum").cast("numpy,float64,HW,L,None")

    def assert_one_image(self):
        imgs = self.to("[image,RGB,RGB]")
        assert len(imgs) == 1

    def first(self, format):
        return self.to(AutoList(format))[0]

    def auto_first(self, format="image,RGB,RGB") -> "AutoImage":
        return self.convert(AutoList(format)).map(lambda x: x[0], format)

    def resize_in_fmt(self, size, fmt="image,RGB,RGB") -> "AutoImage":
        return self.convert(f"[{fmt}]").map(lambda imgs: [i.resize(size) for i in imgs])

    def alpha_as_mask(self):
        self.assert_one_image()
        return self.convert("numpy,float32,CHW,A,0_1").cast("numpy,float32,CHW,L,0_1")

    def show_plot(self, fmt="image,RGB,RGB",show=True):
        from matplotlib import pyplot as plt
        plt.imshow(self.to(fmt))
        if show:
            plt.show()

    def overwrite_with_alpha(self, overlay_with_alpha):
        return AutoImage(overwrite_with_alpha(self, overlay_with_alpha))

    def __repr__(self):
        return f"AutoImage(src={self.src})"

    @staticmethod
    def auto(fmt, value):
        return AutoImage(auto(fmt, value))

    @staticmethod
    def identify(tgt):
        return identify_image(tgt)

    def __getstate__(self):
        return (self.src,)

    def __setstate__(self, state):
        if isinstance(state, tuple):
            self.src = state[0]
        else:
            assert isinstance(state, dict), f"state must be dict:{type(state)}"
            self.src = state['src']

    def image_hash(self):
        data = self.first('jpg_bytes')
        return hashlib.md5(data).hexdigest()


def fetch_img(img: Union[Var, IAutoData, ObjectRef, List, PIL.Image.Image]) -> AutoImage:
    match img:
        case PIL.Image.Image():
            return AutoImage(identify_image(img))
        case (list() | tuple()) as imgs:
            res = []
            for img in imgs:
                res += fetch_img(img).to("[image,RGB,RGB]")
            return AutoImage(auto("[image,RGB,RGB]", res))
        case Var():
            return img.env.put(fetch_img)(img).fetch()
        case AutoImage() as res:
            return res.convert("[image,RGB,RGB]")
        case IAutoData():
            return AutoImage(img)
        case ObjectRef():
            return fetch_img(ray.get(img))
        case _:
            raise RuntimeError(f"unsupported type for fetching:{img}")


def append_imgs(*imgs: Union[Var, IAutoData, ObjectRef, List]) -> AutoImage:
    return fetch_img(list(imgs))


def auto_color_map(img: IAutoData, show=True, value_range="0_1", vmin=None, vmax=None):
    from matplotlib import pyplot as plt
    aim = img.convert(f"numpy,float32,BHWC,RGB,{value_range}")
    # ary = aim.value
    chs = ch_splitter("RGB")
    figs = []
    for i, c in enumerate(chs):
        figs.append(plt.figure())
        tgt = aim.to(f"numpy,float32,HW,{c},{value_range}")
        plt.imshow(tgt, vmin=vmin, vmax=vmax)
        # plt.imshow(aim.to((f"numpy,uint8,HW,{c},0_255")))
        plt.colorbar()
    if show:
        plt.show()
    return figs


def overwrite_with_alpha(a: AutoImage, b: AutoImage):
    fmt = "numpy,float32,BCHW,RGB,0_1"
    rgb = "numpy,float32,BCHW,RGB,0_1"
    alpha = "numpy,float32,BCHW,A,0_1"
    mask = b.to(alpha)
    new_rgba = a.to(rgb) * (1 - mask) + b.to(rgb) * mask
    return AutoImage.auto(fmt, new_rgba)
