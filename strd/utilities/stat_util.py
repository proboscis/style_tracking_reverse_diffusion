from contextlib import contextmanager
from datetime import datetime
from typing import List

import ray
import torch


class GPUMemoryMonitor:
    def __init__(self, device=None):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on your system. Please make sure you have a GPU and the appropriate drivers installed.")

        self.device = device or torch.device("cuda")
        self.start_memory = 0
        self.max_memory_allocated_mb = 0

    def __enter__(self):
        #self.start_memory = torch.cuda.memory_allocated(self.device)
        torch.cuda.reset_max_memory_allocated(self.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.max_memory_allocated_mb = (torch.cuda.max_memory_allocated(self.device) - self.start_memory) / (1024 * 1024)
        return False  # Re-raise exceptions if any


@ray.remote
def log_wandb_run(
        name:str,
        project:str,
        job_type:str,
        config:dict,
        summary:dict,
        history:List[dict],
):
    import wandb
    assert wandb.run is None
    wandb.init(
        name=name,
        project=project,
        job_type=job_type,
        config=config,
    )
    for item in history:
        wandb.log(item)
    wandb.summary.update(summary)
    wandb.finish()


@contextmanager
def monitor_time(name):
    from loguru import logger
    logger.info(f"Running {name}")
    start = datetime.now()
    yield
    end = datetime.now()
    dt = end - start
    logger.info(f"Finished {name} in {(dt).total_seconds()} seconds")
