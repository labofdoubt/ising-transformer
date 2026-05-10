from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor


def patch_to_token(patch: Tensor) -> Tensor:
    flat = patch.to(torch.int64).reshape(*patch.shape[:-2], -1)
    bits = (flat > 0).to(torch.int64)
    shifts = torch.arange(flat.shape[-1], device=patch.device, dtype=torch.int64)
    weights = (2 ** shifts).to(torch.int64)
    return (bits * weights).sum(dim=-1)


def token_to_patch(tokens: Tensor, r: int, c: int) -> Tensor:
    flat = tokens.to(torch.int64).reshape(-1, 1)
    shifts = torch.arange(r * c, device=tokens.device, dtype=torch.int64).view(1, -1)
    bits = ((flat >> shifts) & 1).reshape(*tokens.shape, r, c)
    return bits.to(torch.int64).mul(2).sub(1)


@lru_cache(maxsize=None)
def _all_token_patches_cpu(r: int, c: int) -> Tensor:
    vocab_size = 2 ** (r * c)
    tokens = torch.arange(vocab_size, dtype=torch.int64)
    return token_to_patch(tokens, r, c).contiguous()


def all_token_patches(r: int, c: int, device=None) -> Tensor:
    patches = _all_token_patches_cpu(r, c)
    if device is None:
        return patches
    return patches.to(device=device)
