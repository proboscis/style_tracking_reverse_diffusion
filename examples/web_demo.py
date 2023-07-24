import glob
from pathlib import Path
from typing import List

import gradio as gr
import numpy as np
from loguru import logger

from pinject_design.di.injected import injected_instance, injected_function
from pinject_design.di.util import instances
from pinject_design.run_config_utils import run_main

__meta_design__ = instances(
    default_design_paths=["strd.designs.default_design"]
)

from strd.utilities.auto_image import AutoImage

@injected_function
def web_app_examples(style_examples:list, content_examples:list,/):
    for sty,cnt in zip(style_examples, content_examples):
        yield [
            sty,
            cnt,
            "",
            0.5,
            "all_layers"
        ]


@injected_function
def web_app(
        transfer_style,  # this gets inserted by Design. please check strd/util.py for the impl function.
        web_app_examples,
        /,
        share: bool = True,  # set this to true for a public demo
):
    def _transfer_style(
            style: np.ndarray,
            content: np.ndarray,
            prompt: str = "",
            strength: float = 0.5,
            overrides: str = "all_layers"
    ) -> str:
        style = AutoImage.auto("numpy_rgb", style)
        style = style.resize_in_fmt((512, 512))
        content = AutoImage.auto("numpy_rgb", content)
        content = content.resize_in_fmt((512, 512))
        img: AutoImage = transfer_style(
            style=style,
            content=content,
            prompt=prompt,
            strength=strength,
            overrides=overrides
        )
        return img.to("numpy_rgb")

    demo = gr.Interface(fn=_transfer_style,
                        inputs=[
                            gr.components.Image(type="numpy", label="Style"),
                            gr.components.Image(type="numpy", label="Content"),
                            gr.components.Textbox(label="Prompt"),
                            gr.components.Slider(minimum=0.05, maximum=0.95, label="Strength"),
                            gr.components.Radio(["all_layers", "no_layers"], label="AdaIN target layers")
                        ],
                        outputs=gr.components.Image(type="numpy"),
                        examples=list(web_app_examples())
                        )

    logger.info(f"starting gradio demo. This could take several minutes...")
    demo.launch(share=share, server_name="0.0.0.0")


if __name__ == '__main__':
    run_main()
