from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


def beta_critical(J: float = 1.0) -> float:
    return 0.5 * math.log(1.0 + math.sqrt(2.0)) / J


@dataclass
class ModelConfig:
    L: int
    patch_r: int
    patch_c: int
    hidden_dim: int
    n_heads: int
    n_blocks: int
    use_layernorm: bool
    init_std: float
    use_pos_emb: bool
    use_ap: bool
    beta: float
    J: float = 1.0
    dtype: str = "float32"
    device: str = "cuda"

    def __post_init__(self) -> None:
        assert self.L % self.patch_r == 0
        assert self.L % self.patch_c == 0
        assert self.hidden_dim % self.n_heads == 0
        assert self.patch_area <= 15

    @property
    def patch_area(self) -> int:
        return self.patch_r * self.patch_c

    @property
    def vocab_size(self) -> int:
        return 2 ** self.patch_area

    @property
    def bos_token_id(self) -> int:
        return self.vocab_size

    @property
    def input_vocab_size(self) -> int:
        return self.vocab_size + 1

    @property
    def patch_grid_h(self) -> int:
        return self.L // self.patch_r

    @property
    def patch_grid_w(self) -> int:
        return self.L // self.patch_c

    @property
    def num_patches(self) -> int:
        return self.patch_grid_h * self.patch_grid_w

    @property
    def seq_len(self) -> int:
        return self.num_patches + 1

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.n_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainConfig:
    batch_size: int
    val_batch_size: int
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    learning_rate: float = 1e-3
    total_steps: int = 100_000
    use_cosine_scheduler: bool = False
    validate_every_n: int = 100
    save_logs_every_n: int = 100
    save_checkpoint_every_n: int = 5_000
    resume_checkpoint: str | None = None
    log_dir: str = "runs/default"
    checkpoint_dir: str = "checkpoints/default"
    seed: int = 1234
    grad_clip: float | None = None

    def __post_init__(self) -> None:
        assert self.save_logs_every_n % self.validate_every_n == 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_betas(value: Any) -> tuple[float, float]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    raise ValueError(f"Invalid adam_betas: {value!r}")


def load_config(path: str | Path) -> tuple[ModelConfig, TrainConfig]:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text())
    model_raw = dict(raw["model"])
    train_raw = dict(raw["train"])
    train_raw["adam_betas"] = _coerce_betas(train_raw["adam_betas"])
    model_cfg = ModelConfig(**model_raw)
    train_cfg = TrainConfig(**train_raw)
    return model_cfg, train_cfg
