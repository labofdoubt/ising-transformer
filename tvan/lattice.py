from __future__ import annotations

import torch
from torch import Tensor

from .patches import patch_to_token, token_to_patch


def lattice_to_tokens(spins: Tensor, r: int, c: int) -> Tensor:
    B, L, L2 = spins.shape
    assert L == L2
    grid_h = L // r
    grid_w = L // c
    patches = spins.view(B, grid_h, r, grid_w, c).permute(0, 1, 3, 2, 4).contiguous()
    return patch_to_token(patches).view(B, grid_h * grid_w)


def tokens_to_lattice(tokens: Tensor, L: int, r: int, c: int) -> Tensor:
    B, num_patches = tokens.shape
    grid_h = L // r
    grid_w = L // c
    assert num_patches == grid_h * grid_w
    patches = token_to_patch(tokens, r, c).view(B, grid_h, grid_w, r, c)
    return patches.permute(0, 1, 3, 2, 4).contiguous().view(B, L, L)
