import torch

from tvan.ap import ApproximateProbabilityBias
from tvan.config import ModelConfig
from tvan.lattice import tokens_to_lattice
from tvan.physics import ising_energy


def make_cfg() -> ModelConfig:
    return ModelConfig(
        L=4,
        patch_r=2,
        patch_c=2,
        hidden_dim=16,
        n_heads=4,
        n_blocks=1,
        use_layernorm=True,
        init_std=0.02,
        use_pos_emb=True,
        use_ap=True,
        beta=0.5,
        J=1.0,
        device="cpu",
        dtype="float32",
    )


def test_ap_first_patch_uses_only_internal_energy():
    cfg = make_cfg()
    ap = ApproximateProbabilityBias(cfg)
    bias = ap.bias_for_position(torch.empty(2, 0, dtype=torch.long), patch_pos=0)
    expected = (-cfg.beta * ap.internal_energy).unsqueeze(0).expand(2, -1)
    assert torch.allclose(bias, expected)


def test_ap_left_and_above_terms_are_included():
    cfg = make_cfg()
    ap = ApproximateProbabilityBias(cfg)
    prefix = torch.tensor([[1, 2, 3]], dtype=torch.long)
    patch_pos = 3
    bias = ap.bias_for_position(prefix, patch_pos)

    candidate = 0
    left_token = prefix[0, patch_pos - 1].item()
    above_token = prefix[0, patch_pos - cfg.patch_grid_w].item()
    expected_energy = (
        ap.internal_energy[candidate]
        - (ap.candidate_left_col[candidate].float() * ap.token_right_col[left_token].float()).sum()
        - (ap.candidate_top_row[candidate].float() * ap.token_bottom_row[above_token].float()).sum()
    )
    assert torch.isclose(bias[0, candidate], -cfg.beta * expected_energy)


def test_ap_skips_wraparound_future_neighbors_but_full_energy_keeps_periodic_bonds():
    cfg = make_cfg()
    ap = ApproximateProbabilityBias(cfg)
    prefix = torch.tensor([[1, 2]], dtype=torch.long)
    top_right_pos = 1
    bias = ap.bias_for_position(prefix[:, :top_right_pos], patch_pos=top_right_pos)
    expected = -cfg.beta * (
        ap.internal_energy
        - (ap.candidate_left_col.float() * ap.token_right_col[prefix[0, 0]].float()).sum(dim=1)
    )
    assert torch.allclose(bias[0], expected)

    tokens = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    spins = tokens_to_lattice(tokens, cfg.L, cfg.patch_r, cfg.patch_c).float()
    energy = ising_energy(spins, J=cfg.J)
    manual_wrap = -cfg.J * (spins[:, :, -1] * spins[:, :, 0]).sum(dim=1) - cfg.J * (
        spins[:, -1, :] * spins[:, 0, :]
    ).sum(dim=1)
    assert torch.all(energy.abs() >= manual_wrap.abs())
