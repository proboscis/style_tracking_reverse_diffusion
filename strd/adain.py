from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Mapping, List, Dict

import torch
from cytoolz import valmap
from torch import nn

from strd.utilities.model_flow_visualizer import get_module_in_out, InOut


class IAdainStat(metaclass=ABCMeta):
    @abstractmethod
    def transfer(self, target):
        pass


@dataclass
class AdainStat(IAdainStat):
    mean: torch.Tensor
    std: torch.Tensor

    @staticmethod
    def from_conv2d_out(tensor):
        assert len(tensor.shape) == 4
        return AdainStat(tensor.mean(dim=(2, 3)), tensor.std(dim=(2, 3)))

    def transfer(self, target: torch.Tensor):
        # logger.info(f"transferring adain stat:{self.mean.shape},{self.std.shape}")
        src_mean = target.mean(dim=(2, 3))  # [B,C]
        src_std = target.std(dim=(2, 3))  # [B,C]
        normalized = (target - src_mean[:, :, None, None]) / src_std[:, :, None, None]
        return normalized * self.std[:, :, None, None] + self.mean[:, :, None, None]


class Inputs:
    def __init__(
            self,
            *args, **kwargs
    ):
        self.args = args
        self.kwargs = kwargs


class IAdainStyleTransfer(metaclass=ABCMeta):
    @abstractmethod
    def run(self, style_inputs, target_inputs):
        pass


@dataclass
class AdainStyleTransfer(IAdainStyleTransfer):
    """
    transfers mean/std of style model to the specified target model's conv output.
    style_mapping : key => name of target model layer. val => name of style model layer.
    """
    adain_style_model: nn.Module
    adain_target_model: nn.Module
    style_mapping: Mapping[str, str]

    def __post_init__(self):
        self._last_adain_stats = None
        # raise RuntimeError("this class obsolete.")

    def _replace_conv_forward(self):
        self.replaced_forwards = dict()
        for name, mod in self.adain_target_model.named_modules():
            if name in self.style_mapping:
                self.replaced_forwards[name] = mod.forward
                mod.forward = self._get_replaced_forward(mod, self.style_mapping[name])

    def _restore_conv_forward(self):
        for name, mod in self.adain_target_model.named_modules():
            if name in self.style_mapping:
                mod.forward = self.replaced_forwards[name]

    @property
    def last_adain_stats(self):
        return self._last_adain_stats

    def _get_replaced_forward(self, conv, style_layer_name: str):
        me = self
        conv_forward = conv.forward

        def impl(self, x):
            x = conv_forward(x)
            x = me.last_adain_stats[style_layer_name].transfer(x)
            # todo can we control how much we transfer?
            return x

        return MethodType(impl, conv)

    @staticmethod
    def _gather_adain_stats(model: nn.Module, targets: List[str], *args, **kwargs) -> Dict[str, AdainStat]:
        in_outs: dict = get_module_in_out(model, targets, *args, **kwargs)

        # we assume all outs are shape (B,C,H,W)
        def _get_stat(tensor: InOut):
            return AdainStat.from_conv2d_out(tensor.outputs)

        return valmap(_get_stat, in_outs)

    def gather_stats(self, inputs: Inputs):
        return self._gather_adain_stats(
            self.adain_style_model,
            list(self.style_mapping.values()),
            inputs.args,
            inputs.kwargs,
        )

    def run(self, style_inputs: Inputs, target_inputs: Inputs):
        assert isinstance(style_inputs, Inputs)
        assert isinstance(target_inputs, Inputs)
        self._last_adain_stats = self._gather_adain_stats(
            self.adain_style_model,
            list(self.style_mapping.values()),
            *style_inputs.args,
            **style_inputs.kwargs,
        )
        self._replace_conv_forward()
        res = self.adain_target_model(*target_inputs.args, **target_inputs.kwargs)
        self._restore_conv_forward()
        return res


@dataclass
class GatherableAdainStyleTransfer:
    """
     transfers mean/std of style model to the specified target model's conv output.
     style_mapping : key => name of target model layer. val => name of style model layer.
     """
    adain_style_model: nn.Module
    adain_target_model: nn.Module
    style_mapping: Mapping[str, str]

    def __post_init__(self):
        print(self.style_mapping)

    def _replace_conv_forward(self, adain_stats: dict):
        original_forwards = dict()
        for name, mod in self.adain_target_model.named_modules():
            if name in self.style_mapping:
                original_forwards[name] = mod.forward
                mod.forward = self._get_replaced_forward(mod, adain_stats[self.style_mapping[name]])
        return original_forwards

    def _restore_conv_forward(self, original_forwards):
        for name, mod in self.adain_target_model.named_modules():
            if name in self.style_mapping:
                mod.forward = original_forwards[name]

    def _get_replaced_forward(self, conv, adain_stat: IAdainStat):
        conv_forward = conv.forward

        def impl(self, x):
            x = conv_forward(x)
            x = adain_stat.transfer(x)
            return x

        return MethodType(impl, conv)

    def gather_adain_stats(self, *args, **kwargs) -> Dict[str, AdainStat]:
        in_outs: dict = get_module_in_out(self.adain_style_model, list(self.style_mapping.values()), *args, **kwargs)

        # we assume all outs are shape (B,C,H,W)
        def _get_stat(tensor: InOut):
            return AdainStat.from_conv2d_out(tensor.outputs)

        return valmap(_get_stat, in_outs)

    @contextmanager
    def run_with_adain_stats(self, adain_stats: Dict):
        original_forwards = self._replace_conv_forward(adain_stats)
        yield
        self._restore_conv_forward(original_forwards)

    def transfer(self, stats: Dict[str, IAdainStat], inputs: Inputs):
        with self.run_with_adain_stats(stats):
            return self.adain_target_model(*inputs.args, **inputs.kwargs)

    def run(self, style_inputs: Inputs, target_inputs: Inputs):
        assert isinstance(style_inputs, Inputs)
        assert isinstance(target_inputs, Inputs)
        adain_stats = self.gather_adain_stats(
            *style_inputs.args,
            **style_inputs.kwargs,
        )
        return self.transfer(adain_stats, target_inputs)
