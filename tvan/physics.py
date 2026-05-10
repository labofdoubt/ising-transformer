from __future__ import annotations

import torch
from torch import Tensor


def ising_energy(spins: Tensor, J: float = 1.0) -> Tensor:
    right = torch.roll(spins, shifts=-1, dims=2)
    down = torch.roll(spins, shifts=-1, dims=1)
    return -J * (spins * (right + down)).sum(dim=(1, 2))
