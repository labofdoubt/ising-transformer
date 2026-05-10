from __future__ import annotations

import math

import torch

from .config import beta_critical


def _log_2cosh(x: torch.Tensor) -> torch.Tensor:
    return torch.logaddexp(x, -x)


def _log_2sinh(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log1p(-torch.exp(-2.0 * x))


def exact_ising_free_energy(L: int, beta: float, J: float = 1.0) -> float:
    dtype = torch.float64
    K = torch.tensor(beta * J, dtype=dtype)
    m = torch.arange(0, 2 * L, dtype=dtype)
    cos_term = torch.cos(math.pi * m / L)
    cosh_2k = torch.cosh(2.0 * K)
    sinh_2k = torch.sinh(2.0 * K)
    coth_2k = cosh_2k / sinh_2k
    gamma_arg = cosh_2k * coth_2k - cos_term
    gamma = torch.acosh(gamma_arg)
    odd_gamma = gamma[1::2]
    even_gamma = gamma[0::2]
    odd_x = 0.5 * L * odd_gamma
    even_x = 0.5 * L * even_gamma

    log_z1 = _log_2cosh(odd_x).sum()
    log_z2 = _log_2sinh(odd_x).sum()
    log_z3 = _log_2cosh(even_x).sum()

    if torch.any(even_x <= 0):
        log_z4 = torch.tensor(float("-inf"), dtype=dtype)
    else:
        log_z4 = _log_2sinh(even_x).sum()

    logs = torch.stack([log_z1, log_z2, log_z3])
    sign4 = 1.0 if beta >= beta_critical(J) else -1.0
    max_log = logs.max()
    scaled_sum = torch.exp(logs - max_log).sum()
    if torch.isfinite(log_z4):
        scaled_sum = scaled_sum + sign4 * torch.exp(log_z4 - max_log)
    if scaled_sum <= 0:
        raise ValueError("Finite-size Ising partition function became non-positive.")

    log_prefactor = 0.5 * (L ** 2) * torch.log(2.0 * sinh_2k)
    log_z = math.log(0.5) + log_prefactor + max_log + torch.log(scaled_sum)
    free_energy = -log_z / beta
    return float(free_energy.item())
