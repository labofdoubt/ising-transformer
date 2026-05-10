from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig, TrainConfig


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    step: int,
) -> None:
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": model_cfg.to_dict(),
            "train_config": train_cfg.to_dict(),
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        },
        ckpt_path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    map_location: str | torch.device | None = None,
) -> dict:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload["cuda_rng_state_all"] is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    return payload
