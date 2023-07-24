from dataclasses import dataclass
from typing import Mapping, NamedTuple, Callable

import numpy as np
import torchvision


def dynamic_pix2pix_normalization(tensor):
    """(C,H,W) or (B,C,H,W)"""
    ndim = len(tensor.shape)
    if ndim == 3:
        nc = tensor.shape[0]
    elif ndim == 4:
        nc = tensor.shape[1]
    # logger.debug(tensor.shape)
    if isinstance(tensor,np.ndarray):
        return (tensor-0.5)/0.5
    else:
        return torchvision.transforms.Normalize((0.5,) * nc, (0.5,) * nc)(tensor)


def torch_img_to_p2p_format(state):
    if isinstance(state,Mapping) and state["type"] == "torch":
        return [ConversionEdge(
            dynamic_pix2pix_normalization,
            P2PNormalized(state),
            "pix2pix_normalization",
            1
        )]


class ConversionEdge(NamedTuple):
    f: Callable
    next_state: object
    name: str
    cost: int


@dataclass(unsafe_hash=True)
class P2PNormalized:
    src_format: str

    def __neighbors__(self):
        return ConversionEdge(
            lambda a: a * 0.5 + 0.5,
            self.src_format,
            "inverse pix2pix normalization",
            1
        )
