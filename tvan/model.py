from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig


def _torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float64": torch.float64,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


class IdentityNorm(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


def build_norm(hidden_dim: int, enabled: bool) -> nn.Module:
    return nn.LayerNorm(hidden_dim) if enabled else IdentityNorm()


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.hidden_dim = cfg.hidden_dim
        self.q_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=True)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=True)

    def _split_heads(self, x: Tensor) -> Tensor:
        B, S, D = x.shape
        return x.view(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, x: Tensor) -> Tensor:
        B, H, S, D = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, S, H * D)

    def _sdpa(self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool) -> Tensor:
        if hasattr(F, "scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=is_causal)

        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if is_causal:
            q_len = q.shape[-2]
            k_len = k.shape[-2]
            mask = torch.ones((q_len, k_len), device=q.device, dtype=torch.bool).triu(diagonal=1)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, v)

    def forward(self, x: Tensor) -> Tensor:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        attn = self._sdpa(q, k, v, is_causal=True)
        return self.out_proj(self._merge_heads(attn))

    def forward_step(self, x: Tensor, kv_cache: dict[str, Tensor] | None) -> tuple[Tensor, dict[str, Tensor]]:
        q = self._split_heads(self.q_proj(x))
        k_new = self._split_heads(self.k_proj(x))
        v_new = self._split_heads(self.v_proj(x))
        if kv_cache is None:
            k = k_new
            v = v_new
        else:
            k = torch.cat([kv_cache["k"], k_new], dim=2)
            v = torch.cat([kv_cache["v"], v_new], dim=2)
        attn = self._sdpa(q, k, v, is_causal=False)
        out = self.out_proj(self._merge_heads(attn))
        return out, {"k": k, "v": v}


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.hidden_dim, 4 * cfg.hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(4 * cfg.hidden_dim, cfg.hidden_dim, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln_attn = build_norm(cfg.hidden_dim, cfg.use_layernorm)
        self.ln_mlp = build_norm(cfg.hidden_dim, cfg.use_layernorm)
        self.attn = MultiHeadSelfAttention(cfg)
        self.mlp = FeedForward(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_attn(x))
        x = x + self.mlp(self.ln_mlp(x))
        return x

    def forward_step(self, x: Tensor, kv_cache: dict[str, Tensor] | None) -> tuple[Tensor, dict[str, Tensor]]:
        attn_out, kv_cache = self.attn.forward_step(self.ln_attn(x), kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_mlp(x))
        return x, kv_cache


@dataclass
class ForwardStepOutput:
    logits: Tensor
    kv_cache: list[dict[str, Tensor]]


class TVANTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.compute_dtype = _torch_dtype(cfg.dtype)
        self.autocast_enabled = self.compute_dtype in {torch.float16, torch.bfloat16}
        self.token_embedding = nn.Embedding(cfg.input_vocab_size, cfg.hidden_dim)
        self.pos_embedding = nn.Embedding(cfg.seq_len, cfg.hidden_dim) if cfg.use_pos_emb else None
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_blocks)])
        self.final_norm = build_norm(cfg.hidden_dim, cfg.use_layernorm)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=True)
        self.apply(self._init_weights)

    def _autocast_context(self, device_type: str):
        if not self.autocast_enabled:
            return nullcontext()
        if hasattr(torch, "autocast"):
            return torch.autocast(device_type=device_type, dtype=self.compute_dtype)
        if device_type == "cuda" and hasattr(torch.cuda, "amp"):
            return torch.cuda.amp.autocast(dtype=self.compute_dtype)
        return nullcontext()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def embed_tokens(self, tokens: Tensor) -> Tensor:
        x = self.token_embedding(tokens)
        if self.pos_embedding is not None:
            positions = torch.arange(tokens.shape[1], device=tokens.device)
            x = x + self.pos_embedding(positions)[None, :, :]
        return x

    def embed_step(self, token: Tensor, position: int) -> Tensor:
        x = self.token_embedding(token)
        if self.pos_embedding is not None:
            pos = torch.tensor([position], device=token.device, dtype=torch.long)
            x = x + self.pos_embedding(pos)[None, :, :]
        return x

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.embed_tokens(tokens)
        with self._autocast_context(tokens.device.type):
            for block in self.blocks:
                x = block(x)
            x = self.final_norm(x)
            logits = self.lm_head(x)
        return logits.float()

    def forward_step(
        self,
        last_token: Tensor,
        position: int,
        kv_cache: list[dict[str, Tensor]] | None = None,
    ) -> tuple[Tensor, list[dict[str, Tensor]]]:
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)
        x = self.embed_step(last_token, position)
        new_cache: list[dict[str, Tensor]] = []
        with self._autocast_context(last_token.device.type):
            for block, cache_entry in zip(self.blocks, kv_cache):
                x, cache_entry = block.forward_step(x, cache_entry)
                new_cache.append(cache_entry)
            x = self.final_norm(x)
            logits = self.lm_head(x[:, -1, :])
        return logits.float(), new_cache
