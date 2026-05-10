from __future__ import annotations

import math

import torch

_DTYPE = torch.float64
_DEVICE = torch.device("cpu")


def _validate_exact_ising_inputs(L: int, beta: float, J: float) -> None:
    if L <= 0:
        raise ValueError("L must be positive.")
    if beta <= 0:
        raise ValueError("beta must be positive.")
    if J <= 0:
        raise ValueError("This exact Ising formula assumes ferromagnetic J > 0.")


def _sector_term_constants(K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.tanh(K)
    t2 = t * t
    A = (1.0 + t2) ** 2
    B = 2.0 * t * (1.0 - t2)
    return A, B


def _sector_grids(L: int) -> tuple[torch.Tensor, torch.Tensor]:
    grid = torch.arange(L, dtype=_DTYPE, device=_DEVICE)
    return grid, grid


def _log_omega_sector(
    L: int,
    p_grid: torch.Tensor,
    q_grid: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    mu: float,
    nu: float,
) -> torch.Tensor:
    cos_p = torch.cos(2.0 * math.pi * (p_grid + mu) / L)
    cos_q = torch.cos(2.0 * math.pi * (q_grid + nu) / L)
    term = A - B * (cos_p[:, None] + cos_q[None, :])

    roundoff_tol = 1000.0 * torch.finfo(_DTYPE).eps
    min_term = term.min()
    if min_term < -roundoff_tol:
        raise ValueError(
            f"Encountered a negative Omega factor: min term = {min_term.item()}"
        )

    term = torch.clamp(term, min=0.0)
    return 0.5 * torch.log(term).sum()


def _combine_sector_logs(K_value: float, log_omega: dict[str, torch.Tensor]) -> torch.Tensor:
    positive_sector_logs = torch.stack([
        log_omega["11"],
        log_omega["10"],
        log_omega["01"],
    ])
    log_positive_sum = torch.logsumexp(positive_sector_logs, dim=0)

    critical_K = 0.5 * math.log1p(math.sqrt(2.0))
    critical_tol = 100.0 * torch.finfo(_DTYPE).eps

    if abs(K_value - critical_K) <= critical_tol:
        return log_positive_sum

    if K_value > critical_K:
        return torch.logsumexp(
            torch.stack([
                log_omega["11"],
                log_omega["10"],
                log_omega["01"],
                log_omega["00"],
            ]),
            dim=0,
        )

    log_omega_00 = log_omega["00"]
    if torch.isinf(log_omega_00) and log_omega_00 < 0:
        return log_positive_sum
    if log_omega_00 >= log_positive_sum:
        raise ValueError(
            "Finite-size Ising sector combination became non-positive. "
            f"log_positive_sum={log_positive_sum.item()}, "
            f"log_omega_00={log_omega_00.item()}"
        )
    return log_positive_sum + torch.log1p(-torch.exp(log_omega_00 - log_positive_sum))


def _log_partition_function(L: int, beta: float, K: torch.Tensor, log_sector_sum: torch.Tensor) -> torch.Tensor:
    abs_K = torch.abs(K)
    log_cosh_K = abs_K + torch.log1p(torch.exp(-2.0 * abs_K)) - math.log(2.0)
    return (
        -math.log(2.0)
        + (L * L) * math.log(2.0)
        + 2.0 * (L * L) * log_cosh_K
        + log_sector_sum
    )


def exact_ising_free_energy(L: int, beta: float, J: float = 1.0) -> float:
    """
    Exact finite-L free energy for the isotropic square-lattice Ising model
    on an L x L torus with Hamiltonian

        H = -J sum_{x,y} [
              sigma[x,y] sigma[x+1,y]
            + sigma[x,y] sigma[x,y+1]
        ]

    with periodic boundary conditions. Each nearest-neighbor bond is counted once.

    Returns the total free energy

        F = -(1 / beta) log Z

    assuming k_B = 1.
    """
    _validate_exact_ising_inputs(L, beta, J)
    K_value = beta * J
    K = torch.tensor(K_value, dtype=_DTYPE, device=_DEVICE)
    A, B = _sector_term_constants(K)
    p_grid, q_grid = _sector_grids(L)

    log_omega = {
        "11": _log_omega_sector(L, p_grid, q_grid, A, B, 0.5, 0.5),
        "10": _log_omega_sector(L, p_grid, q_grid, A, B, 0.5, 0.0),
        "01": _log_omega_sector(L, p_grid, q_grid, A, B, 0.0, 0.5),
        "00": _log_omega_sector(L, p_grid, q_grid, A, B, 0.0, 0.0),
    }
    log_sector_sum = _combine_sector_logs(K_value, log_omega)
    log_Z = _log_partition_function(L, beta, K, log_sector_sum)
    return float((-log_Z / beta).item())
