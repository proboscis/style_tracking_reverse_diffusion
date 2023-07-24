import io
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, List, Union

import PIL
import numpy as np
import wandb
from PIL.Image import fromarray
from frozendict import frozendict

from torch import Tensor

from data_tree.dt_wandb.wandb_util import WandbSavedImage
from omni_converter import IAutoData
from omni_converter.solver.rules import RuleEdge
from omni_cv_rules.coconut.convert import ImageDef, TensorLike
from omni_cv_rules.coconut.to_bytes import to_image_bytes
from omni_cv_rules.custom_rules import make_rule
import pampy

from strd.utilities.normalization import dynamic_pix2pix_normalization


@make_rule(WandbSavedImage, "image,RGB,RGB")
def wandb_download_image(t: WandbSavedImage):
    return t.download()


def recursive_py_rule(state, neighbors):
    """
    parse state as python syntax and use this ast as the state
    :param state:
    :return:
    """
    if isinstance(state, str):
        try:
            state = eval(state)
            return neighbors(state)
        except Exception as e:
            pass


COLOR_PATTERN = re.compile("#([0-9a-f]{6})")


# well, how about you add lisp

@make_rule("random_rgb_256x256", "numpy,float32,HWC,RGB,0_1")
def rule_random_rgb_256x256(x):
    return np.random.rand((256, 256, 3)).astype(np.float32),


def parse_color_code(code: str):
    v = int(code, 16)
    r = v >> 16
    g = (v & 0x00ff00) >> 8
    b = (v & 0x0000ff)
    return r, g, b


@make_rule("color_code", "numpy,uint8,HWC,RGBA,0_255")
def rule_color_code(x):
    matched = COLOR_PATTERN.findall(x)
    rgb = parse_color_code(matched[0])
    ary = np.zeros((256, 256, 4), dtype=np.uint8)
    ary[:, :, :3] = rgb
    ary[:, :, 3] = 255
    return ary


@dataclass
class PyImageDef:
    state: Any
    meta: frozendict

    def __hash__(self):
        return hash(self.state) * hash(self.meta)


@dataclass
class Applied:
    src: Any
    kind: Any

    def __hash__(self):
        return hash(self.src) * hash(self.kind)


@dataclass
class Convertable:
    dst: Any
    converter: Callable[[Any], Any]


def _de_pix2pix(x):
    return x * 0.5 + 0.5


def Pix2Pixed(fmt):
    return Convertable(
        fmt,
        _de_pix2pix
    )


def cast_image_def_is_py_image_def(state):
    if isinstance(state, ImageDef):
        return [PyImageDef(state.data_type, state.meta)]
    elif isinstance(state, PyImageDef):
        return [ImageDef(state.state, state.meta)]


def rule_denormalize_normalized(state):
    if isinstance(state, Applied) and state.kind == "pix2pix_normalization":
        return [RuleEdge(
            lambda img: img * 0.5 + 0.5,
            new_format=state.src,
            cost=1,
            name=f"invert application {state.kind}"
        )]


def rule_convert_convertable(state):
    if isinstance(state, Convertable):
        return [RuleEdge(
            state.converter,
            new_format=state.dst,
            cost=1,
            name=f"{state.converter.__name__}"
        )]


def intra_application_cast(state, neighbors: Callable[[Any], List[RuleEdge]]):
    # I want to say that anything castable is castable inside Normalized.
    # but for that we need to
    if isinstance(state, Applied):
        res = []
        for edge in neighbors(state.src):
            if edge.is_cast:
                res.append(RuleEdge(
                    lambda x: x,
                    new_format=Applied(edge.new_format, kind=state.kind),
                    cost=1,
                    name="internal cast",
                    is_cast=True
                ))
        return res


def intra_convertable_cast(state, neighbors: Callable[[Any], List[RuleEdge]]):
    # I want to say that anything castable is castable inside Normalized.
    # but for that we need to
    if isinstance(state, Convertable):
        res = []
        for edge in neighbors(state.dst):
            if edge.is_cast:
                res.append(RuleEdge(
                    lambda x: x,
                    new_format=Convertable(edge.new_format, state.converter),
                    cost=1,
                    name="internal cast",
                    is_cast=True
                ))
        return res


def vgg19_normalize(img: np.ndarray):
    # assume img.shape == B,H,W,C
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (img - mean) / std


def vgg19_denormalize(img: np.ndarray):
    # assume img.shape == B,H,W,C
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return img * std + mean


def rule_prep_to_vgg19(state):
    return pampy.match(state,
                       "numpy,float64,BHWC,RGB,0_1", lambda stat: [RuleEdge(
            converter=vgg19_normalize,
            new_format="numpy,float64,BHWC,RGB,vgg19",
            cost=1,
            name="preprocess image to vgg19 compatible"
        )],
                       "numpy,float64,BHWC,RGB,vgg19", lambda stat: [RuleEdge(
            converter=vgg19_denormalize,
            new_format="numpy,float64,BHWC,RGB,0_1",
            cost=1,
            name="depreprocess image from vgg19 compatible"
        )],
                       Any, lambda stat: []
                       )


def low_high_8bit_to_float_depth(img: np.ndarray):
    # we expect img to be (H,W,C)
    # C == LowBits,HighBits,Org in 8bit
    # output == numpy,float32,HW,L,0_255
    """
    source:
        float lowBits = frac(depth01 * 256);
        float highBits = depth01 - lowBits / 256;
        return float4(lowBits, highBits, depth01, 1);
    """
    assert len(img.shape) == 3
    assert img.dtype == np.uint8
    img = img.astype(np.float32)
    lows = img[:, :, 0] / 256
    highs = img[:, :, 1]
    org = highs + lows
    # ai = auto("numpy,float32,HW,L,0_255",org)
    return org


def rule_24bit_depth_to_float_depth(state):
    if state == "numpy,uint8,HWC,LHD,0_255":
        return [RuleEdge(
            converter=low_high_8bit_to_float_depth,
            new_format="numpy,float32,HW,D,0_255",
            cost=1,
            name="24bit low high depth to float32 depth image"
        )]


def rule_apply_pix2pix(state):
    if isinstance(state, ImageDef):
        if isinstance(state.data_type, TensorLike):
            if state.data_type.arrange == "BCHW":
                return [RuleEdge(
                    converter=lambda x: dynamic_pix2pix_normalization(x),
                    new_format=Applied(state, "pix2pix_normalization"),
                    cost=1,
                    name="apply pix2pix_normalization",
                    is_cast=False
                )]


@make_rule("image,RGB,RGB", "wandb.Image")
def rule_pil_to_wandb_image(img: PIL.Image.Image):
    assert isinstance(img, PIL.Image.Image)
    return wandb.Image(img)


def rule_01_L_to_I16_L(state):
    if state == "numpy,float32,HW,L,0_1":
        return [RuleEdge(
            converter=lambda x: (x * (2 ** 16 - 1)).astype(np.uint16),
            new_format="numpy,uint16,HW,L,0_65535",
            cost=1,
            name="0_1 float to 0_65535 integer",
            is_cast=False
        )]
    elif state == "numpy,uint16,HW,L,0_65535":
        return [
            RuleEdge(
                converter=lambda x: (x / (2 ** 16 - 1)).astype(np.float32),
                new_format="numpy,float32,HW,L,0_1",
                cost=1,
                name="uint16 img to float32 0_1",
                is_cast=False
            ),
            RuleEdge(
                converter=lambda x: PIL.Image.fromarray(x, mode="I;16"),
                new_format="image,I;16",
                cost=1,
                name="0_65535 integer to PIL.Image(I;16)",
                is_cast=False
            )
        ]
    elif state == "image,I;16":
        return [RuleEdge(
            converter=lambda x: np.array(x).astype(np.uint16),
            new_format="numpy,uint32,HW,L,0_65535",
            cost=1,
            name="image,I;16 to numpy",
            is_cast=False
        )]


def rule_pil_l16_to_png_bytes(state):
    if state == "image,I;16":
        return [RuleEdge(
            converter=lambda x: to_image_bytes(x, "png"),
            new_format="png_bytes_i16",
            cost=1,
            name="PIL I;16 to png bytes"
        ), RuleEdge(
            converter=lambda x: to_image_bytes(x, "png"),
            new_format="png_bytes",
            cost=10,
            name="PIL I;16 to png bytes"
        )]


# ok since png_bytes includes a lot of candidate format, we cannot determine the program beforehand.l


@dataclass(frozen=True)
class TensorInfo:
    shape_repr: str
    dtype: str


@dataclass(frozen=True)
class TorchTensor(TensorInfo): pass


@dataclass(frozen=True)
class NumpyTensor(TensorInfo): pass


def rule_add_batch_channel_to_tensor(state):
    match state:
        case TensorInfo(shape_repr, _) as s:
            return [RuleEdge(
                converter=lambda x: x[None],
                new_format=replace(state, shape_repr="B" + shape_repr),
                cost=1,
                name="add batch channel to a tensor"
            )]


def rule_swap_torch_numpy(state):
    match state:
        case TorchTensor(shape_repr, dtype) as s:
            return [RuleEdge(
                converter=lambda x: x.detach().cpu().numpy(),
                new_format=NumpyTensor(shape_repr, dtype=dtype),
                name="torch to numpy"
            )]
        case NumpyTensor(shape_repr, dtype) as s:
            return [RuleEdge(
                converter=lambda x: Tensor(x),
                new_format=TorchTensor(shape_repr, dtype=dtype),
                name="numpy to torch"
            )]


def auto(format, value):
    from strd.utilities.default_factory import _factory
    return _factory(format, value)


def legacy_auto(format):
    def impl(value):
        from strd.utilities.default_factory import _factory
        return _factory(format, value)

    return impl


def identify_image(data: Union[bytes, PIL.Image.Image, IAutoData, WandbSavedImage]) -> "AutoImage":
    from PIL import Image
    from strd.utilities.auto_image import AutoImage

    match data:
        case [PIL.Image.Image() as img, *imgs]:
            fmt = identify_image(img).format
            return AutoImage(auto(f"[{fmt}]", data))
        case PIL.Image.Image():
            mode = data.mode
            return AutoImage(auto(f"image,{mode},{mode}", data))
        case bytes():
            try:
                img = Image.open(io.BytesIO(data))
                return identify_image(img)
            except Exception as e:
                raise ValueError(f"Invalid image format: {e}") from e
        case IAutoData():
            return data
        case WandbSavedImage():
            return AutoImage(auto(WandbSavedImage, data))
        case _:
            raise ValueError(f"could not identify the format of this image:{data}")


def rule_P_to_L(fmt: str):
    if fmt == "image,P,P":
        return [RuleEdge(
            converter=lambda x: x.convert("L"),
            new_format="image,L,L",
            name="P to L",
        )]
