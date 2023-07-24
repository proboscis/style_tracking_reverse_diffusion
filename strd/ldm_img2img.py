import argparse
import os
from contextlib import nullcontext, contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from einops import rearrange
from imwatermark import WatermarkEncoder
from lightning_fabric import seed_everything
from omegaconf import OmegaConf
from torch import autocast
from torchvision.utils import make_grid
from tqdm import trange
from transformers import AutoFeatureExtractor

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSConfig, PLMSSampler
from pinject_design import Injected, injected_function
from strd.utilities.auto_image import AutoImage
from strd.utilities.rulebook import identify_image
from ray_proxy import RemoteInterpreterFactory, IRemoteInterpreter, Var
from stable_diffusion.txt2img import put_watermark, load_model_from_config, numpy_to_pil, load_replacement


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args(args)
    return opt


@dataclass
class Txt2Img:
    opt: argparse.Namespace
    enable_safety_check: bool

    def __post_init__(self):
        import sys
        print(f"python path:{sys.path}")
        opt = self.opt
        os.chdir("../stable-diffusion")
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
        safety_model_id = "CompVis/stable-diffusion-safety-checker"
        if self.enable_safety_check:
            self.safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

        seed_everything(self.opt.seed)
        if self.opt.laion400m:
            print("Falling back to LAION 400M model...")
            self.opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
            self.opt.ckpt = "models/ldm/text2img-large/model.ckpt"
            self.opt.outdir = "outputs/txt2img-samples-laion400m"
        config = OmegaConf.load(f"{self.opt.config}")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.setup_model(config, opt, self.device)
        self.sampler = self.get_sampler(self.model, opt)
        self.batch_size = self.opt.n_samples
        print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
        wm = "StableDiffusionV1"
        self.wm_encoder = WatermarkEncoder()
        self.wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
        self.outpath = self.opt.outdir
        os.makedirs(self.opt.outdir, exist_ok=True)
        self.sample_path = os.path.join(self.outpath, "samples")
        os.makedirs(self.sample_path, exist_ok=True)
        self.base_count = len(os.listdir(self.sample_path))
        self.grid_count = len(os.listdir(self.outpath)) - 1
        self.n_rows = self.opt.n_rows if self.opt.n_rows > 0 else self.batch_size
        print("txt2img model is ready")

    def generate_images(self, prompt, start_code=None):
        precision_scope = autocast if self.opt.precision == "autocast" else nullcontext
        prompt = self.batch_size * [prompt]
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    all_samples = list()
                    for n in trange(self.opt.n_iter, desc="Sampling"):
                        images_torch = self.generate_from_prompt(prompt, start_code)

                        for x_sample in images_torch:
                            sample_img = self.sample_to_img(x_sample)
                            if not self.opt.skip_save:
                                self.save_sample(sample_img)

                        if not self.opt.skip_grid:
                            all_samples.append(images_torch)

                    grid_image = self.to_grid_img(all_samples)
                    if not self.opt.skip_grid:
                        # additionally, save as grid
                        self.save_grid(grid_image)
        print(f"Your samples are ready and waiting for you here: \n{self.outpath} \n"
              f" \nEnjoy.")

    def generate_grid(self, prompt, start_code=None):
        precision_scope = autocast if self.opt.precision == "autocast" else nullcontext
        prompt = self.batch_size * [prompt]
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    all_samples = list()
                    for n in trange(self.opt.n_iter, desc="Sampling"):
                        images_torch = self.generate_from_prompt(prompt, start_code)
                        for x_sample in images_torch:
                            sample_img = self.sample_to_img(x_sample)
                            self.save_sample(sample_img)
                        all_samples.append(images_torch)
                    grid_image = self.to_grid_img(all_samples)
                    self.save_grid(grid_image)
                    return grid_image

    def generate_samples(self, prompt: str, n_samples=None, start_code=None, width=None, height=None):
        n_samples = n_samples or self.batch_size
        prompt = [prompt] * n_samples
        with self.generation_cxt():
            all_samples = list()
            for n in trange(self.opt.n_iter, desc="Sampling"):
                images_torch = self.generate_from_prompt(prompt, start_code, width=width, height=height)
                for x_sample in images_torch:
                    sample_img = self.sample_to_img(x_sample)
                    self.save_sample(sample_img)
                    all_samples.append(sample_img)
        return all_samples

    @contextmanager
    def generation_cxt(self):
        precision_scope = autocast if self.opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    yield

    def prompt_to_generator(self, prompt: str, uc=None):
        prompts = [prompt] * self.batch_size
        if self.opt.scale != 1.0:
            uc = self.model.get_learned_conditioning(self.batch_size * [""])
        c = self.model.get_learned_conditioning(prompts)
        shape = [self.batch_size, self.opt.C, self.opt.H // self.opt.f, self.opt.W // self.opt.f]
        conf = PLMSConfig(
            unconditional_guidance_scale=self.opt.scale,
            unconditional_conditioning=uc,
        )
        yield from self.sampler.sample_generator(
            S=self.opt.ddim_steps,
            batch_size=self.batch_size,
            shape=shape,
            conf=conf,
            conditioning=c,
            eta=self.opt.ddim_eta,
        )

    def generate_from_prompt(self, prompts, start_code=None, uc=None, height=None, width=None):
        H = height or self.opt.H
        W = width or self.opt.W
        if self.opt.fixed_code:
            start_code = torch.randn([len(prompts), self.opt.C, H // self.opt.f, W // self.opt.f],
                                     device=self.device)
        if self.opt.scale != 1.0:
            uc = self.model.get_learned_conditioning(len(prompts) * [""])
        if isinstance(prompts, tuple):
            prompts = list(prompts)
        print(f"trying to get learned conditioning from :{prompts}")
        c = self.model.get_learned_conditioning(prompts)
        shape = [self.opt.C, H // self.opt.f, W // self.opt.f]
        conf = PLMSConfig(
            unconditional_guidance_scale=self.opt.scale,
            unconditional_conditioning=uc,
        )
        samples_ddim, _ = self.sampler.sample(S=self.opt.ddim_steps,
                                              batch_size=len(prompts),
                                              shape=shape,
                                              conf=conf,
                                              conditioning=c,
                                              verbose=False,
                                              eta=self.opt.ddim_eta,
                                              x_T=start_code)
        x_samples_ddim = self.model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        if self.enable_safety_check:
            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
            x_checked_image, has_nsfw_concept = self.check_safety(x_samples_ddim)
            images_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
        else:
            images_torch = x_samples_ddim
        return images_torch

    def save_sample(self, img):
        img.save(os.path.join(self.sample_path, f"{self.base_count:05}.png"))
        self.base_count += 1

    def sample_to_img(self, x_sample):
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        img = Image.fromarray(x_sample.astype(np.uint8))
        img = put_watermark(img, self.wm_encoder)
        return img

    def save_grid(self, img):
        img.save(os.path.join(self.outpath, f'grid-{self.grid_count:04}.png'))
        self.grid_count += 1

    def to_grid_img(self, all_samples):
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=self.n_rows)
        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        img = Image.fromarray(grid.astype(np.uint8))
        img = put_watermark(img, self.wm_encoder)
        return img

    def get_sampler(self, model, opt):
        if self.opt.plms:
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)
        return sampler

    def setup_model(self, config, opt, device):
        model = load_model_from_config(config, f"{self.opt.ckpt}")
        model = model.to(device)
        return model

    def check_safety(self, x_image):
        safety_checker_input = self.safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
        x_checked_image, has_nsfw_concept = self.safety_checker(images=x_image,
                                                                clip_input=safety_checker_input.pixel_values)
        assert x_checked_image.shape[0] == len(has_nsfw_concept)
        for i in range(len(has_nsfw_concept)):
            if has_nsfw_concept[i]:
                x_checked_image[i] = load_replacement(x_checked_image[i])
        return x_checked_image, has_nsfw_concept

    def __repr__(self):
        return f"Txt2Img"


def get_or_create_txt2img_env(remote_interpreter_factory: RemoteInterpreterFactory, name: str):
    rem = remote_interpreter_factory
    import numpy as np
    opt = parse_args(["--plms"])
    env = rem.get_or_create(name, num_gpus=1)
    if "txt2img" not in env:
        txt2img = env.put(Txt2Img)(opt, False)
        txt2img = env.put_named("txt2img", txt2img)
    #env.put(seed_everything)(np.random.randint(0,10000)).fetch()
    return env

@injected_function
def get_or_create_txt2img_env_v2(remote_interpreter_factory: RemoteInterpreterFactory,/, name: str):
    rem = remote_interpreter_factory
    opt = parse_args(["--plms"])
    env = rem.get_or_create(name, num_gpus=1)
    if "txt2img" not in env:
        txt2img = env.put(Txt2Img)(opt, False)
        txt2img = env.put_named("txt2img", txt2img)
    #env.put(seed_everything)(np.random.randint(0,10000)).fetch()
    return env



txt2img_env = Injected.bind(
    get_or_create_txt2img_env,
    name=Injected.pure("txt2img_env")
)


@dataclass
class RemoteTextToImage:
    env: IRemoteInterpreter

    def __post_init__(self):
        self.txt2img: Txt2Img = self.env["txt2img"]

    def generate_samples(self, prompt, n_samples=1, width=1536, height=384) -> Var:
        return self.txt2img.generate_samples(prompt=prompt, n_samples=n_samples, width=width, height=height)

    def generate_auto_samples(self, prompt, n_samples=1, width=512, height=512) -> AutoImage:
        samples = self.generate_samples(prompt, n_samples, width, height)
        return AutoImage(identify_image(samples.fetch()))


remote_txt2img: Injected[RemoteTextToImage] = txt2img_env.map(lambda e: RemoteTextToImage(e))
