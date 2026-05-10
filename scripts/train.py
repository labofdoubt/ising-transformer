from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tvan.ap import ApproximateProbabilityBias
from tvan.checkpoint import load_checkpoint, save_checkpoint
from tvan.config import ModelConfig, TrainConfig, load_config
from tvan.exact_ising import exact_ising_free_energy
from tvan.generation import generate, teacher_forced_log_probs
from tvan.lattice import tokens_to_lattice
from tvan.logging_utils import append_csv, append_jsonl
from tvan.losses import effective_sample_size, free_energy_estimate, score_function_surrogate
from tvan.model import TVANTransformer
from tvan.physics import ising_energy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer: torch.optim.Optimizer, train_cfg: TrainConfig):
    if train_cfg.use_cosine_scheduler:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_cfg.total_steps,
            eta_min=0.0,
        )
    return None


@torch.no_grad()
def run_validation(
    model: TVANTransformer,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    ap_module: ApproximateProbabilityBias | None,
    exact_free_energy: float,
) -> dict[str, float]:
    model.eval()
    patch_tokens, _, spins = generate(model, train_cfg.val_batch_size, model_cfg, ap_module=ap_module, return_spins=True)
    log_q, _ = teacher_forced_log_probs(model, patch_tokens, model_cfg, ap_module=ap_module)
    assert spins is not None
    energy = ising_energy(spins, J=model_cfg.J)
    Fq = free_energy_estimate(log_q, energy, model_cfg.beta)
    ess = effective_sample_size(log_q, energy, model_cfg.beta)
    return {
        "val_ess": float(ess.item()),
        "val_free_energy": float(Fq.item()),
        "val_free_energy_per_spin": float((Fq / (model_cfg.L ** 2)).item()),
        "val_exact_free_energy": float(exact_free_energy),
        "val_exact_free_energy_per_spin": float(exact_free_energy / (model_cfg.L ** 2)),
        "val_free_energy_diff_per_spin": float(((Fq.item() - exact_free_energy) / (model_cfg.L ** 2))),
        "val_free_energy_relative_diff": float((Fq.item() - exact_free_energy) / abs(exact_free_energy)),
        "val_free_energy_relative_abs_diff": float(abs(Fq.item() - exact_free_energy) / abs(exact_free_energy)),
    }


def main() -> None:
    args = parse_args()
    model_cfg, train_cfg = load_config(args.config)
    if args.resume is not None:
        train_cfg.resume_checkpoint = args.resume

    device = torch.device(model_cfg.device)
    seed_everything(train_cfg.seed)

    model = TVANTransformer(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=train_cfg.adam_betas,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = build_scheduler(optimizer, train_cfg)
    ap_module = ApproximateProbabilityBias(model_cfg).to(device) if model_cfg.use_ap else None
    exact_free_energy = exact_ising_free_energy(model_cfg.L, model_cfg.beta, model_cfg.J)

    start_step = 0
    if train_cfg.resume_checkpoint:
        payload = load_checkpoint(train_cfg.resume_checkpoint, model, optimizer, scheduler, map_location=device)
        start_step = int(payload["step"])

    log_dir = Path(train_cfg.log_dir)
    ckpt_dir = Path(train_cfg.checkpoint_dir)
    start_time = time.time()
    last_val = {
        "val_ess": float("nan"),
        "val_free_energy_diff_per_spin": float("nan"),
        "val_free_energy_relative_abs_diff": float("nan"),
    }

    progress = trange(start_step, train_cfg.total_steps, desc="train", dynamic_ncols=True)
    for step in progress:
        model.eval()
        with torch.no_grad():
            patch_tokens, _, _ = generate(model, train_cfg.batch_size, model_cfg, ap_module=ap_module, return_spins=False)

        model.train()
        log_q, _ = teacher_forced_log_probs(model, patch_tokens, model_cfg, ap_module=ap_module)
        spins = tokens_to_lattice(patch_tokens, model_cfg.L, model_cfg.patch_r, model_cfg.patch_c)
        energy = ising_energy(spins, J=model_cfg.J)
        surrogate_loss, Fq = score_function_surrogate(log_q, energy, model_cfg.beta)
        train_free_energy_relative_abs_diff = abs(Fq.item() - exact_free_energy) / abs(exact_free_energy)

        optimizer.zero_grad(set_to_none=True)
        surrogate_loss.backward()
        if train_cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if (step + 1) % train_cfg.validate_every_n == 0:
            last_val = run_validation(model, model_cfg, train_cfg, ap_module, exact_free_energy)

        lr = optimizer.param_groups[0]["lr"]
        progress.set_postfix(
            step=f"{step + 1}/{train_cfg.total_steps}",
            free_energy=f"{Fq.item():.4f}",
            train_free_energy_relative_abs_diff=f"{train_free_energy_relative_abs_diff:.3e}",
            validation_ess=f"{last_val['val_ess']:.4f}",
            validation_free_energy_diff=f"{last_val['val_free_energy_diff_per_spin']:.3e}",
            lr=f"{lr:.3e}",
        )

        if (step + 1) % train_cfg.save_logs_every_n == 0:
            record = {
                "step": step + 1,
                "lr": lr,
                "train_free_energy": float(Fq.item()),
                "train_free_energy_per_spin": float(Fq.item() / (model_cfg.L ** 2)),
                "train_exact_free_energy": float(exact_free_energy),
                "train_exact_free_energy_per_spin": float(exact_free_energy / (model_cfg.L ** 2)),
                "train_free_energy_relative_abs_diff": float(train_free_energy_relative_abs_diff),
                "val_ess": last_val["val_ess"],
                "val_free_energy_diff_per_spin": last_val["val_free_energy_diff_per_spin"],
                "val_free_energy_relative_diff": last_val.get("val_free_energy_relative_diff", float("nan")),
                "val_free_energy_relative_abs_diff": last_val.get("val_free_energy_relative_abs_diff", float("nan")),
                "time_sec": time.time() - start_time,
            }
            append_jsonl(log_dir / "metrics.jsonl", record)
            append_csv(log_dir / "metrics.csv", record)

        if (step + 1) % train_cfg.save_checkpoint_every_n == 0:
            save_checkpoint(
                ckpt_dir / f"step_{step + 1}.pt",
                model,
                optimizer,
                scheduler,
                model_cfg,
                train_cfg,
                step + 1,
            )


if __name__ == "__main__":
    main()
