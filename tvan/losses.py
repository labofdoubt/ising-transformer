from __future__ import annotations

import math

import torch
from torch import Tensor


def score_function_surrogate(log_q: Tensor, energy: Tensor, beta: float) -> tuple[Tensor, Tensor]:
    S = log_q + beta * energy
    Fq_estimate = S.mean() / beta
    baseline = S.mean().detach()
    surrogate_loss = ((S.detach() - baseline) * log_q).mean() / beta
    return surrogate_loss, Fq_estimate


def free_energy_estimate(log_q: Tensor, energy: Tensor, beta: float) -> Tensor:
    return (log_q + beta * energy).mean() / beta


def effective_sample_size(log_q: Tensor, energy: Tensor, beta: float) -> Tensor:
    B = log_q.shape[0]
    log_w = -beta * energy - log_q
    log_mean_w = torch.logsumexp(log_w, dim=0) - math.log(B)
    log_mean_w2 = torch.logsumexp(2.0 * log_w, dim=0) - math.log(B)
    return torch.exp(2.0 * log_mean_w - log_mean_w2)
