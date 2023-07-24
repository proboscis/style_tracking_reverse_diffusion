
# Installation

```
git clone https://github.com/proboscis/style_tracking_reverse_diffusion.git/
cd style_tracking_reverse_diffusion
make install
```

# Introduction

## Link to original paper

# Tutorial
## Run on local machine
```bash
python examples/main.py transfer_style -overrides='{"img2img_sd_ckpt_path":"../style_tracking_reverse_diffusion/sd-v1-4.ckpt"}' - \
	--style example_data/image_pairs/style_0.png --content example_data/image_pairs/content_0.png \
	- to '"image,RGB,RGB"' - save '/root/host_home/result.png'
```
Please change img2img_sd_ckpt_path to the path of the sd-v1-4.ckpt file, relative to the stable_diffsion repository or absolute path.

## Try Web Demo
```bash
python examples/web_demo.py web_app -overrides='{"img2img_sd_ckpt_path":"../style_tracking_reverse_diffusion/sd-v1-4.ckpt"}'
```

## Dependency
This library depends on the dependency injection called [pinject_design](https://github.com/proboscis/pinject-design).

TODO add screen shots

### Docker
### Local

# TODO
- [ ] make this use huggingface model instead of the stable-diffusion repo.