from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tvan.ap import ApproximateProbabilityBias
from tvan.checkpoint import load_checkpoint
from tvan.config import load_config
from tvan.generation import generate
from tvan.model import TVANTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    model_cfg, _ = load_config(args.config)
    device = torch.device(model_cfg.device)
    model = TVANTransformer(model_cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    ap_module = ApproximateProbabilityBias(model_cfg).to(device) if model_cfg.use_ap else None
    patch_tokens, log_q, spins = generate(model, args.batch_size, model_cfg, ap_module=ap_module, return_spins=True)
    np.savez(
        args.output,
        patch_tokens=patch_tokens.cpu().numpy(),
        log_q=log_q.cpu().numpy(),
        spins=spins.cpu().numpy() if spins is not None else None,
    )


if __name__ == "__main__":
    main()
