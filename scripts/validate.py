from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tvan.ap import ApproximateProbabilityBias
from tvan.checkpoint import load_checkpoint
from tvan.config import load_config
from tvan.exact_ising import exact_ising_free_energy
from tvan.generation import generate, teacher_forced_log_probs
from tvan.logging_utils import append_jsonl
from tvan.losses import effective_sample_size, free_energy_estimate
from tvan.model import TVANTransformer
from tvan.physics import ising_energy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    model_cfg, train_cfg = load_config(args.config)
    device = torch.device(model_cfg.device)
    batch_size = args.batch_size or train_cfg.val_batch_size
    num_samples = args.num_samples or train_cfg.val_batch_size

    model = TVANTransformer(model_cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    ap_module = ApproximateProbabilityBias(model_cfg).to(device) if model_cfg.use_ap else None

    all_log_q = []
    all_energy = []
    steps = math.ceil(num_samples / batch_size)
    for _ in range(steps):
        current_batch = min(batch_size, num_samples - len(all_log_q) * batch_size)
        patch_tokens, log_q_gen, spins = generate(model, current_batch, model_cfg, ap_module=ap_module, return_spins=True)
        log_q, _ = teacher_forced_log_probs(model, patch_tokens, model_cfg, ap_module=ap_module)
        if log_q_gen.shape == log_q.shape:
            max_diff = (log_q_gen - log_q).abs().max().item()
            print(f"log_q_cache_vs_teacher_max_diff={max_diff:.6e}")
        all_log_q.append(log_q.cpu())
        assert spins is not None
        all_energy.append(ising_energy(spins, J=model_cfg.J).cpu())

    log_q = torch.cat(all_log_q, dim=0)
    energy = torch.cat(all_energy, dim=0)
    Fq = free_energy_estimate(log_q, energy, model_cfg.beta).item()
    ess = effective_sample_size(log_q, energy, model_cfg.beta).item()
    F_exact = exact_ising_free_energy(model_cfg.L, model_cfg.beta, model_cfg.J)
    metrics = {
        "num_samples": int(log_q.shape[0]),
        "free_energy": Fq,
        "free_energy_per_spin": Fq / (model_cfg.L ** 2),
        "exact_free_energy": F_exact,
        "exact_free_energy_per_spin": F_exact / (model_cfg.L ** 2),
        "free_energy_diff_per_spin": (Fq - F_exact) / (model_cfg.L ** 2),
        "free_energy_relative_diff": (Fq - F_exact) / abs(F_exact),
        "ess": ess,
    }
    for key, value in metrics.items():
        print(f"{key}={value}")
    if args.log_file:
        append_jsonl(args.log_file, metrics)


if __name__ == "__main__":
    main()
