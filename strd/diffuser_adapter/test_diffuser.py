import loguru
from diffusers import DDPMPipeline

import ray
from loguru import logger

from pinject_design.di.util import instances, providers


# %%
def provide_ray(ray_head_address, ray_namespace, ray_env):
    if not ray.is_initialized():
        logger.info(f"initializing ray with address={ray_head_address}, namespace={ray_namespace}")
        ray.init(
            address=ray_head_address,
            namespace=ray_namespace,
            runtime_env=ray_env
        )
        logger.info("ray initialized")
    return ray


ray_design = instances(
    ray_address="raycluster-kuberay-head-svc",
    ray_namespace="test",
    ray_env={
        "pip": ['diffusers','torch']
    }
) + providers(
    ray_head_address=lambda ray_address: f"ray://{ray_address}:10001",
    ray=provide_ray,
)

# %%

rg = (ray_design
      + instances(logger=loguru.logger)
      ).to_graph()
ray = rg['ray']
@ray.remote(
    num_gpus=1,
)
def generate():
    ddpm = DDPMPipeline.from_pretrained("google/ddpm-cat-256").to("cuda")
    image = ddpm(num_inference_steps=25).images[0]
    image
# %%
# TODO enable gpu support on the worker node.
img = ray.get(generate.remote())
# %%