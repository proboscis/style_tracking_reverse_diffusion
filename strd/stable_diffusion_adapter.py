"""make variations of input image"""

import argparse
import os
from contextlib import nullcontext, contextmanager
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Sequence, List

import PIL
import numpy as np
import torch
from PIL import Image
from einops import rearrange, repeat
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.utils import make_grid
from tqdm import tqdm

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.diffusionmodules.util import noise_like
from ldm.util import instantiate_from_config
from pinject_design import Design, injected_function
from ray_proxy import RemoteInterpreterFactory
from strd.ddim_encoding import DdimEncodingHistoryProvider


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def prep_image(path_or_img):
    if isinstance(path_or_img, str):
        img = Image.open(path_or_img)
    else:
        img = path_or_img

    image = img.convert("RGB")
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def parse(args: Sequence[str]):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
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
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
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
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
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
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
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


def find_stable_diffusion_dir() -> Path:
    import ldm.util
    return Path(ldm.util.__file__).parent.parent


def serve_img2img(working_dir, argv):
    if not working_dir:
        working_dir = find_stable_diffusion_dir()
    print(f"chainging dir to {working_dir} for stable diffusion")
    os.chdir(working_dir)
    opt = parse(argv)
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    my_sampler = MyDDIMSampler(sampler)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    precision_scope = autocast if opt.precision == "autocast" else nullcontext

    def get_enc_of_img(img: PIL.Image.Image, strength):
        init_image = prep_image(img).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(strength * opt.ddim_steps)
        print(f"target t_enc is {t_enc} steps")
        return init_latent, t_enc

    def decode(samples):
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        return x_samples

    @contextmanager
    def work_scope():
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    yield

    def img2img_gen(img, prompt, strength):
        prompts = batch_size * [prompt]
        init_latent, t_enc = get_enc_of_img(img, strength)
        with work_scope():
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(batch_size * [""])
            c = model.get_learned_conditioning(prompts)

            # encode (scaled latent)
            z_enc = sampler.stochastic_encode(init_latent,
                                              torch.tensor([t_enc] * batch_size).to(device))
            # decode it
            for x_dec in my_sampler.decode_generator(
                    z_enc, c, t_enc,
                    unconditional_guidance_scale=opt.scale,
                    unconditional_conditioning=uc, ):
                yield x_dec

    def img2img(img, prompt: str, strength):
        prompts = batch_size * [prompt]
        init_latent, t_enc = get_enc_of_img(img, strength)
        with work_scope():
            all_samples = list()
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(batch_size * [""])
            c = model.get_learned_conditioning(prompts)

            # encode (scaled latent)
            z_enc = sampler.stochastic_encode(init_latent,
                                              torch.tensor([t_enc] * batch_size).to(device))
            # decode it
            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                     unconditional_conditioning=uc, )
            x_samples = decode(samples)
            all_samples.append(x_samples)

            # additionally, save as grid
            grid = torch.stack(all_samples, 0)
            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid = make_grid(grid, nrow=n_rows)

            # to image
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            img = Image.fromarray(grid.astype(np.uint8))
            return dict(
                grid=img,
                samples=x_samples
            )

    sampler.make_schedule(
        ddim_num_steps=opt.ddim_steps,
        ddim_eta=opt.ddim_eta,
        verbose=False
    )
    design = Design(
        classes=[
            DdimEncodingHistoryProvider
        ]
    ).bind_instance(
        model=model,
        sampler=sampler,
        get_enc_of_img=get_enc_of_img,
        precision_scope=precision_scope,
        opt=opt,
        batch_size=batch_size,
        device=device
    )
    graph = design.to_graph()
    # we can use this graph's session to do anything.

    return {**globals(), **locals()}

@injected_function
def get_annon_img2img_env(
        remote_interpreter_factory: RemoteInterpreterFactory,
        img2img_sd_working_dir: str,
        img2img_sd_argv: List[str],
        /,
):
    env = remote_interpreter_factory.create(num_gpus=1)
    env["img2img_vars"] = env.put(serve_img2img)(img2img_sd_working_dir, img2img_sd_argv)
    # this assertion actually waits for the remote function to finish
    assert 'img2img_vars' in env, 'img2img_vars not in env'
    return env


@dataclass
class MyDDIMSampler:
    sampler: DDIMSampler

    @torch.no_grad()
    def p_sample_ddim_and_noise(self, x, c, t, index, unconditional_guidance_scale, unconditional_conditioning):
        src = self.sampler
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = src.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = src.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        alphas = src.ddim_alphas
        alphas_prev = src.ddim_alphas_prev
        sqrt_one_minus_alphas = src.ddim_sqrt_one_minus_alphas
        sigmas = src.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        temperature = 1.0

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, False) * temperature
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0, e_t

    @torch.no_grad()
    def decode_generator(self,
                         x_latent, cond, t_start,
                         unconditional_guidance_scale=1.0,
                         unconditional_conditioning=None
                         ):
        timesteps = self.sampler.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, x_0_hat, e_t = self.p_sample_ddim_and_noise(x_dec, cond, ts, index=index,
                                                               unconditional_guidance_scale=unconditional_guidance_scale,
                                                               unconditional_conditioning=unconditional_conditioning)

            yield dict(
                x_0_hat=x_0_hat,
                x_dec=x_dec,
                e_t=e_t,
                i=i,
                step=step
            )
