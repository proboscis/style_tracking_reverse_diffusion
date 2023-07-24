from omni_converter import AutoRuleBook
from strd.utilities.normalization import torch_img_to_p2p_format
from strd.utilities.rulebook import wandb_download_image, rule_random_rgb_256x256, rule_color_code, rule_prep_to_vgg19, \
    rule_pil_to_wandb_image, rule_denormalize_normalized, rule_convert_convertable, rule_apply_pix2pix, \
    rule_01_L_to_I16_L, rule_pil_l16_to_png_bytes, rule_add_batch_channel_to_tensor, rule_swap_torch_numpy, \
    cast_image_def_is_py_image_def, intra_application_cast, intra_convertable_cast, recursive_py_rule

ARCHPAINTER_RULES = AutoRuleBook().add_rules(
    torch_img_to_p2p_format,
    wandb_download_image,
    rule_random_rgb_256x256,
    rule_color_code,
    rule_prep_to_vgg19,
    rule_pil_to_wandb_image,
    rule_denormalize_normalized,
    rule_convert_convertable,
    rule_apply_pix2pix,
    rule_01_L_to_I16_L,
    rule_pil_l16_to_png_bytes,
    rule_add_batch_channel_to_tensor,
    rule_swap_torch_numpy,
).add_cast(
    cast_image_def_is_py_image_def, "ImageDef == PyImageDef"
).add_alias(
    "numpy,float64,BHWC,RGB,vgg19", "numpy,float64,BHWC,RGB,imagenet"
).add_recursive_rule(
    intra_application_cast,
).add_recursive_rule(
    intra_convertable_cast
).add_recursive_rule(
    recursive_py_rule
).set_id("ARCHPAINTER_RULES")
