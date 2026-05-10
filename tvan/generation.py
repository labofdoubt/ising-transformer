from __future__ import annotations

import torch
from torch import Tensor

from .ap import ApproximateProbabilityBias
from .config import ModelConfig
from .lattice import tokens_to_lattice
from .model import TVANTransformer


def prepend_bos(patch_tokens: Tensor, bos_token_id: int) -> Tensor:
    bos = torch.full((patch_tokens.shape[0], 1), bos_token_id, dtype=torch.long, device=patch_tokens.device)
    return torch.cat([bos, patch_tokens], dim=1)


def generate(
    model: TVANTransformer,
    batch_size: int,
    cfg: ModelConfig,
    ap_module: ApproximateProbabilityBias | None = None,
    return_spins: bool = True,
) -> tuple[Tensor, Tensor, Tensor | None]:
    device = torch.device(cfg.device)
    tokens = torch.full((batch_size, 1), cfg.bos_token_id, dtype=torch.long, device=device)
    kv_cache = None
    log_q = torch.zeros(batch_size, device=device)

    for patch_pos in range(cfg.num_patches):
        logits, kv_cache = model.forward_step(tokens[:, -1:], position=patch_pos, kv_cache=kv_cache)
        if cfg.use_ap and ap_module is not None:
            logits = logits + ap_module.bias_for_position(tokens[:, 1:], patch_pos)
        log_probs = torch.log_softmax(logits, dim=-1)
        sampled = torch.distributions.Categorical(logits=logits).sample()
        log_q = log_q + log_probs.gather(1, sampled[:, None]).squeeze(1)
        tokens = torch.cat([tokens, sampled[:, None]], dim=1)

    patch_tokens = tokens[:, 1:]
    spins = tokens_to_lattice(patch_tokens, cfg.L, cfg.patch_r, cfg.patch_c) if return_spins else None
    return patch_tokens, log_q, spins


def teacher_forced_log_probs(
    model: TVANTransformer,
    patch_tokens: Tensor,
    cfg: ModelConfig,
    ap_module: ApproximateProbabilityBias | None = None,
) -> tuple[Tensor, Tensor]:
    inputs = prepend_bos(patch_tokens, cfg.bos_token_id)
    logits = model(inputs)[:, :-1, :]
    if cfg.use_ap and ap_module is not None:
        logits = logits + ap_module.bias_for_sequence(patch_tokens)
    log_probs = torch.log_softmax(logits, dim=-1)
    selected = log_probs.gather(dim=-1, index=patch_tokens.unsqueeze(-1)).squeeze(-1)
    return selected.sum(dim=1), logits
