import torch

from tvan.config import ModelConfig
from tvan.generation import prepend_bos
from tvan.model import TVANTransformer


def make_cfg() -> ModelConfig:
    return ModelConfig(
        L=4,
        patch_r=2,
        patch_c=2,
        hidden_dim=16,
        n_heads=4,
        n_blocks=2,
        use_layernorm=True,
        init_std=0.02,
        use_pos_emb=True,
        use_ap=False,
        beta=0.3,
        J=1.0,
        device="cpu",
        dtype="float32",
    )


def test_causal_masking():
    torch.manual_seed(0)
    cfg = make_cfg()
    model = TVANTransformer(cfg)
    a = torch.tensor([[cfg.bos_token_id, 1, 2, 3, 4]])
    b = torch.tensor([[cfg.bos_token_id, 1, 2, 7, 6]])
    logits_a = model(a)
    logits_b = model(b)
    assert torch.allclose(logits_a[:, :3, :], logits_b[:, :3, :], atol=1e-6, rtol=1e-6)


def test_kv_cache_matches_full_forward():
    torch.manual_seed(1)
    cfg = make_cfg()
    model = TVANTransformer(cfg)
    patch_tokens = torch.tensor([[1, 2, 3, 0]])
    inputs = prepend_bos(patch_tokens, cfg.bos_token_id)
    full_logits = model(inputs)

    kv_cache = None
    cached_logits = []
    for position in range(cfg.num_patches):
        logits, kv_cache = model.forward_step(inputs[:, position : position + 1], position=position, kv_cache=kv_cache)
        cached_logits.append(logits)

    cached_logits = torch.stack(cached_logits, dim=1)
    assert torch.allclose(cached_logits, full_logits[:, :-1, :], atol=1e-6, rtol=1e-6)
