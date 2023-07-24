from pinject_design import Injected
from pinject_design.run_config_utils import load_variable_by_module_path, find_default_design_paths, pinject_main, \
    run_main
from strd.util import transfer_style

__default_design_paths__ = ['strd.designs.default_design']


def run(target: str, design_path=None):
    from loguru import logger
    known_targets = globals()
    design_path = find_default_design_paths(__file__, design_path)[0]
    logger.info(f"Using design path: {design_path}")
    design = load_variable_by_module_path(design_path)
    if target in known_targets:
        target = known_targets[target]
    else:
        target = load_variable_by_module_path(target)
    return design.provide(target)


example_32: Injected = transfer_style(
    style="example_data/image_pairs/style_3.png",
    content="example_data/image_pairs/content_2.png",
).show_plot()
# This can be run by
# python examples/main.py example_32

# this is equivalent to:
# python examples/main.py transfer_style - --style example_data/image_pairs/style_3.png --content example_data/image_pairs/content_2.png - show_plot

if __name__ == '__main__':
    run_main()
