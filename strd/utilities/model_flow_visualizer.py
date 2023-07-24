import builtins
from dataclasses import dataclass
from typing import Union, List, Any

import ipywidgets as widgets
import numpy as np
import torch
from IPython.core.display import display
from ipywidgets import interactive, Label
from matplotlib import pyplot as plt
from rx import operators as ops
from rx.subject import Subject, ReplaySubject
from torch import Tensor, nn
from tqdm import tqdm

from data_tree import series
from strd.utilities.rulebook import auto
from ray_proxy import Var


def gen_hook(name, sub):
    def _hook(_self, input, output):
        sub.on_next((name, _self, input, output))

    return _hook


def get_call_hook(module):
    sub = Subject()
    for name, m in module.named_modules():
        m.register_forward_hook(gen_hook(name, sub))
    return sub


def infer_widget(x):
    match x:
        case Var():
            return infer_remote_widget(x)
        case np.ndarray():
            return ary_to_widget(x)
        case torch.Tensor():
            return ary_to_widget(x.cpu().detach().numpy())
        case tuple() | list():
            return widgets.VBox([
                infer_widget(item) for item in x
            ])
        case dict():
            return widgets.VBox([
                widgets.VBox([widgets.Text(k), infer_widget(v)]) for k, v in x.items()
            ])
        case None:
            return widgets.Label("None")
        case _:
            raise RuntimeError(f"unknown value type for infer_widget:{type(x)},{x}")


def infer_remote_widget(x: Var) -> widgets.Widget:
    match x.__type__():
        case np.ndarray:
            return remote_ary_to_widget(x)
        case torch.Tensor:
            return remote_ary_to_widget(x)
        case builtins.tuple | builtins.list:
            return widgets.VBox([infer_remote_widget(x) for x in x])
        case builtins.dict:
            return widgets.VBox([
                widgets.VBox([widgets.Text(k.fetch()), infer_remote_widget(v)]) for k, v in x.items()
            ])
        case tp if tp == type(None):
            return widgets.Label("None")

    raise RuntimeError(f"unknown value type for infer_remote_widget:{x.__type__()}")


def download_proxy(item):
    import builtins
    match item:
        case Var():
            return download_proxy(item.fetch())
        case dict():
            return {k: download_proxy(v) for k, v in item.items()}
        case list():
            return [download_proxy(v) for v in item]
        case tuple():
            return tuple(download_proxy(v) for v in item)
        case _:
            return item


def to_detached_cpu(item):
    match item:
        case torch.Tensor():
            return item.detach().cpu()
        case dict():
            return {k: to_detached_cpu(v) for k, v in item.items()}
        case list():
            return [to_detached_cpu(v) for v in item]
        case tuple():
            return tuple(to_detached_cpu(v) for v in item)
        case _:
            return item


def ary_to_widget(x: np.ndarray):
    # we need accordion when the input is a batch of many channels
    """
        case (b, 3, h, w):
            img = auto("numpy,float32,BCHW,RGB,None", x)
            return img.to("widget")
        case (b, 4, h, w):
            img = auto("numpy,float32,BCHW,RGBA,None", x)
            return img.to("widget")
        case (b, 1, h, w):
            img = auto("numpy,float32,BCHW,L,None", x)
            return img.to("widget")
    """
    match x.shape:
        case (b, c, h, w):
            # we have a lot of info, so lets make it choosable
            def _widget(i):
                data = x[i]

                return display(auto("numpy,float32,BHW,L,None", data).to("widget"))

            return interactive(_widget, i=(0, len(x) - 1, 1))
        case shape:
            return Label(f"unknown shape:{shape}")


def plot_tensor_stat(x: Union[Var, Tensor, np.ndarray]):
    match x:
        case Var() if x.__type__() == torch.Tensor:
            ary = x.flatten()[:10000].detach().cpu().numpy().fetch()
        case Var() if x.__type__() == np.ndarray:
            ary = x.flatten()[:10000].fetch()
        case Tensor():
            ary = x.flatten()[:10000].detach().cpu().numpy()
        case np.ndarray():
            ary = x.flatten()[:10000]
        case _:
            raise RuntimeError(f"can't plot tensor:{x}'")
    plt.hist(ary, bins=100)
    plt.show()


def remote_ary_to_widget(x: Var):
    rtype = x.__type__()
    assert rtype in {np.ndarray, torch.Tensor}, "remote type must be Tensor"
    if rtype == np.ndarray:
        type_str = "numpy"
    elif rtype == torch.Tensor:
        type_str = "torch"

    match x.shape.fetch():
        case (b, c, h, w):
            def _widget(i):
                data = x[i]
                plot_tensor_stat(data)
                display(x.env.put(auto)(f"{type_str},float32,BHW,L,None", data).to("image,RGB,RGB").fetch())

            return interactive(_widget, i=(0, len(x) - 1, 1))
        case (b,c): #flattened
           def _widget(i):
               data = x[i]
               plot_tensor_stat(data)
           return interactive(_widget, i=(0, len(x) - 1, 1))

        case shape:
            return Label(f"unknown shape:{shape}")


def module_in_outs(model, *args, **kwargs):
    assert isinstance(model, torch.nn.Module)
    replay = ReplaySubject()
    hook = get_call_hook(model)
    _unsub = hook.subscribe(replay)
    res = model(*args, **kwargs)
    replay.on_completed()
    _unsub.dispose()
    outputs = replay.pipe(ops.to_list()).run()
    return outputs


def get_module_names(model: torch.nn.Module, *args, **kwargs):
    assert isinstance(model, torch.nn.Module)
    call_history = []

    def gen_hook(name):
        def hook(self, _input, _output):
            call_history.append(name)

        return hook

    handles = []
    for name, m in model.named_modules():
        hook = gen_hook(name)
        handle = m.register_forward_hook(hook)
        handles.append(handle)
    res = model(*args, **kwargs)
    for handle in handles:
        handle.remove()
    return call_history


def get_module_in_out(model: torch.nn.Module, module_names: List[str], *args, **kwargs):
    mods = dict(model.named_modules())
    results = dict()

    def gen_hook(name):
        def hook(self, _input, _output):
            results[name] = InOut(name, _input, _output)

        return hook

    handles = []
    try:
        for name in module_names:
            handles.append(mods[name].register_forward_hook(gen_hook(name)))
    except KeyError as e:
        raise RuntimeError(f"module {e} not found in model with modules:{mods.keys()}") from e

    res = model(*args, **kwargs)
    for handle in handles:
        handle.remove()
    return results


def visualize_module_flow(model, *args, **kwargs):
    assert isinstance(model, torch.nn.Module) or (
            isinstance(model, Var) and issubclass(model.__type__(), torch.nn.Module))
    if isinstance(model, Var):
        mod_names = [n.fetch() for n in model.env.put(get_module_names)(model, *args, **kwargs)]
    else:
        mod_names = get_module_names(model, *args, **kwargs)
    return get_module_flow_widget(model, mod_names, *args, **kwargs)


def get_module_flow_widget(model, module_names, *args, **kwargs):
    vizs = []
    for name in module_names:
        load_button = widgets.Button(description="press to show")
        plot_out = widgets.Output()

        def gen_on_click(_name, _plot_out):
            def on_click(*_args, **_kwargs):
                if isinstance(model, Var):
                    inout = model.env.put(get_module_in_out)(model, [_name], *args, **kwargs)[_name]
                else:
                    inout = get_module_in_out(model, [_name], *args, **kwargs)[_name]
                in_widget, out_widget = infer_widget(inout.inputs), infer_widget(inout.outputs)
                with _plot_out:
                    display("fetching data...")
                    display(widgets.HBox([in_widget, out_widget]))
                    display(f"input shape:{inout.inputs[0].shape}")
                    display(f"output shape:{inout.outputs.shape}")

            return on_click

        load_button.on_click(gen_on_click(name, plot_out))
        vizs.append(widgets.VBox([
            load_button,
            plot_out
        ]))
    acc = widgets.Accordion(children=vizs, titles=module_names)
    for i, title in enumerate(module_names):
        acc.set_title(i, title)
    return acc


@dataclass
class InOut:
    module_name: str
    inputs: Any
    outputs: Any

    def _ipython_display_(self):
        self.display()

    def display(self):
        in_widget, out_widget = infer_widget(self.inputs), infer_widget(self.outputs)
        display(f"module: {self.module_name}")
        display(widgets.HBox([in_widget, out_widget]))
        display(f"input shape:{self.inputs[0].shape}")
        display(f"output shape:{self.outputs.shape}")

    @staticmethod
    def from_remote(rp: Var):
        return InOut(
            rp.module_name.fetch(),
            inputs=rp.inputs,
            outputs=rp.outputs
        )


@dataclass
class ModelFlowVisualizer:
    # TODO move this to public repo
    def __init__(self, target, *args, **kwargs):
        self.target: Union[torch.nn.Module, Var] = target
        self.model_args: tuple = args
        self.model_kwargs: dict = kwargs
        if isinstance(self.target, Var):
            self.module_names = self.target.env.put(get_module_names)(self.target, *self.model_args,
                                                                      **self.model_kwargs).fetch()
        else:
            self.module_names = get_module_names(self.target, *self.model_args, **self.model_kwargs)

    def get_widget(self):
        return get_module_flow_widget(self.target, self.module_names, *self.model_args, **self.model_kwargs)

    def get_outputs(self, mod_names: List[str]) -> dict:
        assert isinstance(mod_names, list)
        model = self.target
        args = self.model_args
        kwargs = self.model_kwargs
        if isinstance(model, Var):
            ios = model.env.put(get_module_in_out)(model, mod_names, *args, **kwargs)
            return {k.fetch(): InOut.from_remote(v) for k, v in ios.items()}
        else:
            return get_module_in_out(model, mod_names, *args, **kwargs)

    def in_out_series(self):
        return series(self.module_names).map(
            lambda n: self.get_outputs([n])[n]
        )

    def _ipython_display_(self):
        display(self.get_widget())


def visualize_module_in_outs(
        in_outs,
        prog_bar=tqdm,
):
    children = []
    titles = []
    for name, m, _in, _out in prog_bar(in_outs):
        assert _out is not None
        # I think I need to make this lazy
        load_button = widgets.Button(description="press to show")
        plot_out = widgets.Output()

        def on_click(*args, **kwargs):
            in_widget, out_widget = infer_widget(_in), infer_widget(_out)
            with plot_out:
                display("fetching data...")
                display(widgets.HBox([in_widget, out_widget]))
                display(f"input shape:{_in[0].shape}")
                display(f"output shape:{_out.shape}")

        load_button.on_click(on_click)
        children.append(widgets.VBox([
            load_button,
            plot_out
        ]))
        if isinstance(name, Var):
            name = name.fetch()
        titles.append(name)

    acc = widgets.Accordion(children=children, titles=titles)
    for i, title in prog_bar(enumerate(titles)):
        acc.set_title(i, title)
    return acc


class ModelFlowVisualizerDeprecated:

    def show_flow(self, model: nn.Module, args, kwargs, prog_bar=tqdm):
        outputs = module_in_outs(model, args, kwargs)
        return visualize_module_in_outs(outputs, prog_bar)
