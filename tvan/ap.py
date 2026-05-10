from __future__ import annotations

import torch
from torch import Tensor

from .config import ModelConfig
from .patches import all_token_patches


class ApproximateProbabilityBias:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.all_patches = all_token_patches(cfg.patch_r, cfg.patch_c)
        self.internal_energy = self._compute_internal_energy(self.all_patches, cfg.J)
        self.candidate_left_col = self.all_patches[:, :, 0].contiguous()
        self.candidate_top_row = self.all_patches[:, 0, :].contiguous()
        self.token_right_col = self.all_patches[:, :, -1].contiguous()
        self.token_bottom_row = self.all_patches[:, -1, :].contiguous()

    @staticmethod
    def _compute_internal_energy(patches: Tensor, J: float) -> Tensor:
        energy = torch.zeros(patches.shape[0], dtype=torch.float32)
        if patches.shape[2] > 1:
            energy = energy - J * (patches[:, :, :-1] * patches[:, :, 1:]).sum(dim=(1, 2)).float()
        if patches.shape[1] > 1:
            energy = energy - J * (patches[:, :-1, :] * patches[:, 1:, :]).sum(dim=(1, 2)).float()
        return energy

    def to(self, device: torch.device | str) -> "ApproximateProbabilityBias":
        device = torch.device(device)
        other = ApproximateProbabilityBias.__new__(ApproximateProbabilityBias)
        other.cfg = self.cfg
        other.vocab_size = self.vocab_size
        other.all_patches = self.all_patches.to(device=device)
        other.internal_energy = self.internal_energy.to(device=device)
        other.candidate_left_col = self.candidate_left_col.to(device=device)
        other.candidate_top_row = self.candidate_top_row.to(device=device)
        other.token_right_col = self.token_right_col.to(device=device)
        other.token_bottom_row = self.token_bottom_row.to(device=device)
        return other

    def bias_for_position(self, prefix_patch_tokens: Tensor, patch_pos: int) -> Tensor:
        device = prefix_patch_tokens.device
        if self.internal_energy.device != device:
            ap = self.to(device)
            return ap.bias_for_position(prefix_patch_tokens, patch_pos)

        B = prefix_patch_tokens.shape[0]
        C = self.cfg.patch_grid_w
        pr = patch_pos // C
        pc = patch_pos % C

        bias = (-self.cfg.beta * self.internal_energy).unsqueeze(0).expand(B, -1).clone()

        if pc > 0:
            left_tokens = prefix_patch_tokens[:, patch_pos - 1]
            left_edge = self.token_right_col[left_tokens].float()
            left_term = -self.cfg.J * (
                self.candidate_left_col.float().unsqueeze(0) * left_edge.unsqueeze(1)
            ).sum(dim=2)
            bias = bias + (-self.cfg.beta * left_term)

        if pr > 0:
            above_tokens = prefix_patch_tokens[:, patch_pos - C]
            above_edge = self.token_bottom_row[above_tokens].float()
            above_term = -self.cfg.J * (
                self.candidate_top_row.float().unsqueeze(0) * above_edge.unsqueeze(1)
            ).sum(dim=2)
            bias = bias + (-self.cfg.beta * above_term)

        return bias

    def bias_for_sequence(self, patch_tokens: Tensor) -> Tensor:
        B, N = patch_tokens.shape
        all_biases = []
        empty_prefix = patch_tokens[:, :0]
        for patch_pos in range(N):
            prefix = empty_prefix if patch_pos == 0 else patch_tokens[:, :patch_pos]
            all_biases.append(self.bias_for_position(prefix, patch_pos))
        return torch.stack(all_biases, dim=1)
