import glob
import os
from pathlib import Path
from typing import Any

import loguru
import ray

from pinject_design import Design, Injected
from pinject_design.di.graph import IObjectGraph
from pinject_design.di.injected import injected_instance, injected_function
from pinject_design.di.util import providers, classes, instances
from ray_proxy import RemoteInterpreterFactory, Var, IRemoteInterpreter
from ray_proxy.injected_resource import InjectedResource
from ray_proxy.remote_env import ActorRefRemoteInterpreter
from ray_proxy.resource_design import ResourceDesign
from strd.adain_diffusion import SimplifiedImageToImage, LatentSpaceDecoder
from strd.adain_diffusion_resource import additional_design_for_adain_diffusion
from strd.adain_options import adain_options
from strd.distributed_adain import adain_img2img_decoded
from strd.stable_diffusion_adapter import get_annon_img2img_env
from strd.util import transfer_style
from strd.utilities.pickling import to_pickled


@injected_instance
def resource__gpu_env(session: IObjectGraph):
    factory = session.factory

    @Injected.bind
    def impl(gpu):
        rem = factory.create()[RemoteInterpreterFactory]
        env = rem.create(num_gpus=1)
        devices: str = env.put(lambda: os.environ["CUDA_VISIBLE_DEVICES"])().fetch()
        gpu_ids = env.put(lambda: ray.get_gpu_ids())().fetch()
        if len(devices.split(',')) != 1:
            ray.kill(env.remote_env)
            raise RuntimeError(f"cuda device not allocated for this env!:{devices},gpu_ids={gpu_ids}")
        return env

    return impl


@injected_instance
def resource__img2img_env(session):
    design = session.design

    @Injected.bind
    def resource(gpu):
        return design.to_graph()[get_annon_img2img_env]()

    return resource


@Injected.bind
def img2img_session(img2img_env) -> Var[IObjectGraph]:
    return img2img_env["img2img_vars"]["graph"].child_session()


def kill_env(env: IRemoteInterpreter):
    assert isinstance(env, ActorRefRemoteInterpreter)
    ray.kill(env.remote_env)


def to_resource(res: Injected[Injected[Any]], scope: str, num_issuable: int, destructor=None):
    return res.map(lambda x: InjectedResource(x, scope, num_issuable, destructor=destructor))


@injected_function
def make_resource(res: Injected[Any], scope, num_issuable: int, destructor=None):
    return InjectedResource(res, scope, num_issuable, destructor=destructor)


@Injected.bind
def img2img_decode(img2img_env: Var[IRemoteInterpreter]):
    r_d = img2img_env["img2img_vars"]["design"] + additional_design_for_adain_diffusion
    r_g = r_d.to_graph()
    return r_g[LatentSpaceDecoder]


@Injected.bind
def adain_img2img(img2img_env: Var[IRemoteInterpreter]):
    env = img2img_env
    design = env["img2img_vars"]["design"] + additional_design_for_adain_diffusion
    ddim_sampler = design.provide(SimplifiedImageToImage)
    return ddim_sampler


@injected_instance
def cluster_design(
        make_resource,
        ray_num_gpus_in_cluster,
        resource__img2img_env,
        resource__gpu_env,
):
    return ResourceDesign().bind_provider(
        img2img_env=make_resource(resource__img2img_env, "reserved", ray_num_gpus_in_cluster, kill_env),
        img2img_session=InjectedResource(img2img_session, "ondemand", ray_num_gpus_in_cluster),
        adain_img2img=InjectedResource(adain_img2img, "ondemand", ray_num_gpus_in_cluster),
        gpu=InjectedResource(Injected.pure("gpu_token"), "reserved", ray_num_gpus_in_cluster, ),
        gpu_env=make_resource(resource__gpu_env, "ondemand", ray_num_gpus_in_cluster, kill_env),
    )


def provide_ray(logger, ray_head_address, ray_namespace, ray_env):
    if not ray.is_initialized():
        logger.info(f"initializing ray with address={ray_head_address}, namespace={ray_namespace}")
        ray.init(
            address=ray_head_address,
            namespace=ray_namespace,
            runtime_env=ray_env,
        )
    return ray


def provide_local_ray():
    if not ray.is_initialized():
        ray.init()
    return ray


ray_design_for_cluster = Design(
).bind_instance(
    ray_namespace="archpainter",
).bind_provider(
    ray_head_address=lambda ray_address: f"ray://{ray_address}:10001",
    ray_job_submission_address=lambda ray_address: f"http://{ray_address}:8265",
)

ray_design_for_publish = providers(
    ray=provide_local_ray,
)

pickled_adain_options = to_pickled(Injected.pure(adain_options), Path("adain_options.pkled")).eval()
adain_options_all_layers = pickled_adain_options.map(
    lambda opts: {l: l for l in opts['all']}
)


@Injected.bind
def adain_img2img_overrides(adain_options_all_layers: Injected):
    return Design().bind_provider(
        style_mapping=adain_options_all_layers,
    )


def provide_img2img_argv(img2img_sd_ckpt_path: Path):
    # ["--ckpt", "the path of stable diffusion ckpt"]
    return ["--ckpt", str(img2img_sd_ckpt_path)]


img2img_env_design = instances(
    img2img_sd_ckpt_path=Path("../stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt"),
) + providers(
    img2img_sd_argv=provide_img2img_argv,
    img2img_sd_working_dir=None
)


@injected_instance
def ray_num_gpus_in_cluster(ray):
    return ray.cluster_resources().get("GPU", 0)


default_design = Design() + instances(
) + providers(
    adain_options_all_layers=adain_options_all_layers,
    cluster_design=cluster_design,
    sch=lambda cluster_design, ray: cluster_design.to_scheduler(ray=ray),
    ray=provide_ray,
    logger=lambda: loguru.logger,
    adain_img2img_decoded=adain_img2img_decoded,
    transfer_style=transfer_style,
    style_examples=lambda: glob.glob("example_data/image_pairs/style_*.png"),
    content_examples=lambda: glob.glob("example_data/image_pairs/content_*.png"),
) + classes(
    remote_interpreter_factory=RemoteInterpreterFactory
) + ray_design_for_cluster + img2img_env_design + ray_design_for_publish
