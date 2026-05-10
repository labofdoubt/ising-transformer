# Coding-agent instruction: build a transparent PyTorch repo reproducing tVAN-style transformer sampling for 2D Ising lattices

Build a clean PyTorch repository that implements a transformer-based Variational Autoregressive Network for 2D spin systems, following the attached paper’s main ideas: represent a lattice as a sequence of spin patches, generate patches autoregressively, train by minimizing variational free energy, optionally add approximate-probability energy biases, validate with ESS and exact Ising free energy.

The implementation must be from scratch. Do not import nanoGPT or other transformer repos. PyTorch, tqdm, numpy, yaml/json logging are allowed.

## 1. Repository structure

Use this structure:

```text
tvan/
  __init__.py
  config.py              # dataclass configs and validation
  patches.py             # patch <-> integer token mapping
  lattice.py             # full lattice <-> token sequence
  physics.py             # Ising energy, optional EA couplings
  exact_ising.py         # finite periodic Ising free energy
  model.py               # transformer blocks
  ap.py                  # approximate probability energy bias
  generation.py          # autoregressive generation with KV cache
  losses.py              # free energy estimator, score-function surrogate, ESS
  checkpoint.py          # save/load checkpoints and RNG state
  logging_utils.py       # jsonl/csv logging helpers
scripts/
  train.py
  validate.py
  sample.py
configs/
  ising_l32_2x4.yaml
  ising_l120_3x4_ap.yaml
tests/
  test_patches.py
  test_lattice_roundtrip.py
  test_energy.py
  test_exact_ising.py
  test_causal_and_cache.py
  test_ap.py
```

Prefer small, testable functions over monolithic scripts.

## 2. Configs

Create two config dataclasses.

### `ModelConfig`

Required fields:

```python
L: int                         # lattice length, L x L
patch_r: int                   # patch height r
patch_c: int                   # patch width c
hidden_dim: int                # transformer hidden dimension d
n_heads: int
n_blocks: int
use_layernorm: bool
init_std: float                # normal init std for Linear/Embedding
use_pos_emb: bool
use_ap: bool
beta: float                    # default critical beta for J=1
J: float = 1.0                 # scalar Ising coupling
dtype: str = "float32"         # optionally bf16 for model forward
device: str = "cuda"
```

Derived fields:

```python
patch_area = patch_r * patch_c
vocab_size = 2 ** patch_area
bos_token_id = vocab_size
input_vocab_size = vocab_size + 1
patch_grid_h = L // patch_r
patch_grid_w = L // patch_c
num_patches = patch_grid_h * patch_grid_w
seq_len = num_patches + 1      # includes BOS
```

Validation rules:

```python
assert L % patch_r == 0
assert L % patch_c == 0
assert hidden_dim % n_heads == 0
assert patch_area <= 15        # warn or error; vocab grows exponentially
```

Use default critical inverse temperature

\[
\beta_c = \frac{1}{2J}\log(1+\sqrt{2}).
\]

For `J=1`, this is approximately `0.4406867935`.

### `TrainConfig`

Required fields:

```python
batch_size: int
val_batch_size: int
adam_betas: tuple[float, float]     # default (0.9, 0.95)
weight_decay: float
learning_rate: float
total_steps: int
use_cosine_scheduler: bool
validate_every_n: int
save_logs_every_n: int
save_checkpoint_every_n: int
resume_checkpoint: str | None
log_dir: str
checkpoint_dir: str
seed: int
grad_clip: float | None = None
```

Validate:

```python
assert save_logs_every_n % validate_every_n == 0
```

The progress bar must show:

```text
step / total_steps
free_energy
validation_ess
validation_free_energy_diff
lr
```

## 3. Patch-to-token mapping

Implement `patches.py`.

A patch is an `r x c` array of spins in `{−1, +1}`. Flatten it row-major. Map spins to bits using:

```python
-1 -> 0
+1 -> 1
```

Use the paper’s little-endian convention:

\[
\text{token} = \sum_{k=0}^{rc-1} b_k 2^k,
\]

where `b_k` is the bit at flattened index `k`.

So for a `2 x 4` patch:

```text
token 0  -> all -1
token 1  -> first flattened spin +1, rest -1
token 2  -> second flattened spin +1, rest -1
token 4  -> third flattened spin +1, rest -1
```

Implement:

```python
def patch_to_token(patch: Tensor) -> Tensor:
    """
    patch shape: [..., r, c], values -1 or +1
    returns: token shape [...]
    """

def token_to_patch(tokens: Tensor, r: int, c: int) -> Tensor:
    """
    tokens shape [...]
    returns: [..., r, c], values -1 or +1
    """

def all_token_patches(r: int, c: int, device=None) -> Tensor:
    """
    returns tensor [vocab_size, r, c] with every possible patch.
    Cache this because AP needs it frequently.
    """
```

Use integer operations for mapping. Tests must verify roundtrip for all tokens when `r*c <= 12`.

## 4. Full lattice encoder/decoder

Implement `lattice.py`.

Given a lattice `spins` of shape `[B, L, L]`, divide it into non-overlapping patches of shape `r x c` in row-major patch order:

```python
patch_row = 0 ... L//r - 1
patch_col = 0 ... L//c - 1
patch_index = patch_row * patch_grid_w + patch_col
```

Implement:

```python
def lattice_to_tokens(spins: Tensor, r: int, c: int) -> Tensor:
    """
    spins: [B, L, L], values -1/+1
    returns patch_tokens: [B, num_patches]
    """

def tokens_to_lattice(tokens: Tensor, L: int, r: int, c: int) -> Tensor:
    """
    tokens: [B, num_patches]
    returns spins: [B, L, L], values -1/+1
    """
```

Do not include BOS in these functions. BOS is a transformer input concern.

## 5. Transformer architecture

Implement `model.py`.

The model is decoder-only and autoregressive.

### Inputs and outputs

Input sequence:

```text
[BOS, t_1, t_2, ..., t_N]
```

where `N = num_patches`.

The BOS token is passed through the transformer, but it is never a target. Logits at position `0`, i.e. the BOS position, predict `t_1`. Logits at position `i` predict `t_{i+1}`.

Output logits shape:

```python
[B, S, vocab_size]
```

The output vocabulary excludes BOS. BOS can be embedded but cannot be predicted.

For teacher-forced training with a full sequence:

```python
input_tokens = concat([bos], patch_tokens)   # [B, N+1]
logits = model(input_tokens)                 # [B, N+1, vocab_size]
prediction_logits = logits[:, :-1, :]        # [B, N, vocab_size]
targets = patch_tokens                       # [B, N]
```

Ignore `logits[:, -1, :]`.

### Embedding

```python
token_embedding = nn.Embedding(vocab_size + 1, hidden_dim)
```

If `use_pos_emb=True`, add:

```python
pos_embedding = nn.Embedding(num_patches + 1, hidden_dim)
```

If `use_pos_emb=False`, skip positional embedding entirely.

### Transformer block

Use pre-norm blocks:

\[
x \leftarrow x + \text{Attention}(\text{LN}(x))
\]

\[
x \leftarrow x + \text{MLP}(\text{LN}(x))
\]

If `use_layernorm=False`, every LayerNorm becomes identity, including the final LayerNorm.

The MLP is:

```python
Linear(d, 4d)
GELU
Linear(4d, d)
```

Use biased linear layers. Initialize all linear and embedding weights from:

\[
\mathcal{N}(0, \text{init\_std}^2)
\]

Initialize all biases to zero.

Final unembedding:

```python
lm_head = nn.Linear(hidden_dim, vocab_size, bias=True)
```

Do not weight-tie because the input vocabulary includes BOS but output vocabulary does not.

### Attention

Implement multi-head causal self-attention from scratch.

For full-sequence training, use PyTorch’s:

```python
torch.nn.functional.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True,
)
```

This will use FlashAttention when the installed PyTorch/CUDA backend supports it.

Use shapes:

```python
q, k, v: [B, n_heads, S, head_dim]
head_dim = hidden_dim // n_heads
```

Dropout should default to zero.

## 6. KV-cached autoregressive generation

Implement generation in `generation.py`.

Generation starts from BOS and produces exactly `num_patches` patch tokens.

Algorithm:

```python
tokens = [[bos_token_id]] repeated batch_size times
kv_cache = empty cache for every transformer block
log_q = zeros [B]

for patch_pos in range(num_patches):
    logits, kv_cache = model.forward_step(
        last_token=tokens[:, -1:],
        position=patch_pos,
        kv_cache=kv_cache,
    )
    # logits shape [B, vocab_size], predicting patch_pos

    if use_ap:
        logits = logits + ap_bias_for_position(
            prefix_patch_tokens=tokens[:, 1:],
            patch_pos=patch_pos,
        )

    probs = softmax(logits, dim=-1)
    sampled = Categorical(probs).sample()
    log_q += log_softmax(logits, dim=-1).gather(sampled)

    append sampled to tokens
```

Important attention detail:

For cached single-token decoding, call scaled-dot-product attention with `is_causal=False`, because the cache already contains only past and current keys. Using `is_causal=True` with query length `1` and key length `>1` can incorrectly mask keys.

Return:

```python
patch_tokens: [B, num_patches]
log_q: [B]
spins: [B, L, L] optional
```

Add a test verifying that cached generation logits match non-cached forward logits for the same prefix.

## 7. Approximate probabilities, AP

Implement `ap.py`.

AP adds a physical energy bias to the transformer logits:

\[
q(t^i_j \mid t^{<i}) =
\operatorname{softmax}_j\left(f_j(t^{<i}) - \beta E^i(t_j)\right).
\]

In this repo, because BOS predicts the first patch, the same formula applies to the BOS-position logits. For patch position `i=0`, `E^0` includes only internal patch bonds.

### Patch geometry

Let:

```python
R = L // r
C = L // c
patch_pos = pr * C + pc
pr = patch_pos // C
pc = patch_pos % C
```

Patch `patch_pos` covers lattice rows:

```python
pr*r : (pr+1)*r
```

and columns:

```python
pc*c : (pc+1)*c
```

### Full Ising energy convention

Use one and only one convention everywhere:

\[
E(s) = -J \sum_{x=0}^{L-1}\sum_{y=0}^{L-1}
s_{x,y}\left(s_{(x+1)\bmod L,y} + s_{x,(y+1)\bmod L}\right).
\]

This counts every nearest-neighbor bond exactly once. There are `2 * L * L` bonds.

### AP energy for a candidate patch

For each candidate token `j`, decode it to a candidate patch `P_j`.

Compute:

\[
E^i(t_j) =
- J \sum_{\text{internal bonds}} s_a s_b
- J \sum_{\text{causal external bonds}} s_a s_b.
\]

Internal bonds:

```text
within candidate patch:
  horizontal bonds: P[row, col] -- P[row, col+1], col < c-1
  vertical bonds:   P[row, col] -- P[row+1, col], row < r-1
```

External causal bonds are only with already-generated neighboring patches. With row-major generation, these are:

```text
left neighbor, if pc > 0:
  current P[row, 0] interacts with left_patch[row, c-1]

above neighbor, if pr > 0:
  current P[0, col] interacts with above_patch[r-1, col]
```

Periodic boundary condition matters for the full energy, but AP must not use future tokens. Therefore, do not include wraparound AP bonds to the rightmost patch when `pc == 0`, and do not include wraparound AP bonds to the bottom row when `pr == 0`, because those neighboring patches have not yet been generated.

A robust general rule is:

```python
include a neighbor bond in AP iff the neighbor patch index < current patch index
```

Always include all periodic bonds in the full energy used for training and validation.

### Efficient AP implementation

Precompute:

```python
all_patches: [V, r, c]     # V = 2 ** (r*c)
internal_energy: [V]
```

For scalar Ising `J`, internal energy is the same for every patch position.

For boundary terms, avoid building huge `[num_patches, V, V]` tensors. Either:

1. For `V <= 4096`, optionally precompute two matrices:

```python
left_boundary_energy[left_token, candidate_token]  -> scalar
above_boundary_energy[above_token, candidate_token] -> scalar
```

2. Otherwise compute on the fly using vectorized boundary spin arrays:

```python
candidate_left_col: [V, r]
candidate_top_row: [V, c]
token_right_col: [V, r]
token_bottom_row: [V, c]
```

For a batch at patch position `i`:

```python
bias = -beta * internal_energy[None, :]  # [B, V]

if pc > 0:
    left_tokens = generated_tokens[:, i - 1]
    E_left = -J * sum_rows(
        candidate_left_col[None, :, :] * token_right_col[left_tokens][:, None, :]
    )
    bias += -beta * E_left

if pr > 0:
    above_tokens = generated_tokens[:, i - C]
    E_above = -J * sum_cols(
        candidate_top_row[None, :, :] * token_bottom_row[above_tokens][:, None, :]
    )
    bias += -beta * E_above
```

Return AP bias with shape `[B, vocab_size]`.

For teacher-forced full forward, compute AP bias for every predicted position `0..num_patches-1` and add it to `logits[:, :-1, :]`.

## 8. Physics energy

Implement `physics.py`.

### Ising energy

```python
def ising_energy(spins: Tensor, J: float = 1.0) -> Tensor:
    """
    spins: [B, L, L], values -1/+1
    returns E: [B]
    periodic boundary conditions
    counts each nearest-neighbor bond once
    """
    right = torch.roll(spins, shifts=-1, dims=2)
    down = torch.roll(spins, shifts=-1, dims=1)
    return -J * (spins * (right + down)).sum(dim=(1, 2))
```

This must match AP and exact free energy conventions.

Optionally support Edwards-Anderson later:

\[
E(s) =
-\sum_{x,y} J^{\text{right}}_{x,y}s_{x,y}s_{x,y+1}
-\sum_{x,y} J^{\text{down}}_{x,y}s_{x,y}s_{x+1,y}.
\]

But keep Ising scalar `J=1` as the primary target.

## 9. Exact finite periodic Ising free energy (I think this formula is wrong – I had to fix it later in code)

Implement `exact_ising.py` in float64.

Use this for validation of square `L x L` Ising with periodic boundaries and scalar coupling `J`.

Let:

\[
K = \beta J.
\]

Define:

\[
\cosh \gamma_m =
\cosh(2K)\coth(2K) - \cos\left(\frac{\pi m}{L}\right),
\qquad m=0,\ldots,2L-1.
\]

Then define:

\[
\log Z_1 = \sum_{k=0}^{L-1}
\log\left(2\cosh\left(\frac{L\gamma_{2k+1}}{2}\right)\right),
\]

\[
\log Z_2 = \sum_{k=0}^{L-1}
\log\left(2\sinh\left(\frac{L\gamma_{2k+1}}{2}\right)\right),
\]

\[
\log Z_3 = \sum_{k=0}^{L-1}
\log\left(2\cosh\left(\frac{L\gamma_{2k}}{2}\right)\right),
\]

\[
\log Z_4 = \sum_{k=0}^{L-1}
\log\left(2\sinh\left(\frac{L\gamma_{2k}}{2}\right)\right).
\]

For `K >= Kc`, use:

\[
Z =
\frac12
\left(2\sinh(2K)\right)^{L^2/2}
\left(Z_1 + Z_2 + Z_3 + Z_4\right).
\]

For `K < Kc`, the conventional finite-torus formula uses a minus sign on the fourth term:

\[
Z =
\frac12
\left(2\sinh(2K)\right)^{L^2/2}
\left(Z_1 + Z_2 + Z_3 - Z_4\right).
\]

At criticality, \(\gamma_0=0\), so \(Z_4=0\) and the sign is irrelevant.

Compute with log-sum-exp, and for the signed high-temperature case handle the subtraction carefully. Return:

\[
F_{\text{exact}} = -\frac{1}{\beta}\log Z.
\]

Add regression tests:

```python
beta_c = 0.5 * log(1 + sqrt(2))

exact_free_energy(L=120, beta=beta_c, J=1) / 120**2
# should be approximately -2.10975198

exact_free_energy(L=128, beta=beta_c, J=1) / 128**2
# should be approximately -2.10973977
```

These numbers match the attached paper’s reported finite-size values.

## 10. Free energy “loss” and gradient estimator

Implement `losses.py`.

The target Boltzmann distribution is:

\[
p(s) = \frac{1}{Z}e^{-\beta E(s)}.
\]

The model distribution is autoregressive:

\[
q_\theta(t_1,\ldots,t_N)
=
\prod_{i=1}^{N} q_\theta(t_i \mid t_{<i}).
\]

With BOS, this becomes:

\[
q_\theta(t_1,\ldots,t_N)
=
\prod_{i=0}^{N-1} q_\theta(t_{i+1} \mid \text{BOS},t_{\le i}).
\]

The variational free energy is:

\[
F_q =
\frac{1}{\beta}
\mathbb{E}_{s\sim q_\theta}
\left[
\log q_\theta(s) + \beta E(s)
\right].
\]

For a generated batch:

```python
patch_tokens: [B, N]
spins: [B, L, L]
log_q: [B]
energy: [B]
S = log_q + beta * energy
Fq = S.mean() / beta
```

Important: the scalar `Fq` is what should be logged as the free-energy estimate. But do not backpropagate through `Fq` naively, because samples were generated discretely from the model.

Use the score-function gradient estimator from the paper:

\[
\nabla_\theta F_q =
\frac{1}{\beta}
\mathbb{E}_{q_\theta}
\left[
(S - \mathbb{E}_{q_\theta}[S])
\nabla_\theta \log q_\theta(s)
\right],
\]

where:

\[
S = \log q_\theta(s)+\beta E(s).
\]

Implement the backward surrogate:

```python
def score_function_surrogate(log_q: Tensor, energy: Tensor, beta: float) -> tuple[Tensor, Tensor]:
    """
    log_q has gradient.
    energy is treated as fixed.
    returns:
      surrogate_loss for backward
      Fq_estimate for logging
    """
    S = log_q + beta * energy
    Fq_estimate = S.mean() / beta

    baseline = S.mean().detach()
    surrogate_loss = ((S.detach() - baseline) * log_q).mean() / beta

    return surrogate_loss, Fq_estimate
```

Call:

```python
surrogate_loss.backward()
optimizer.step()
```

The model minimizes the variational free energy by following this gradient.

## 11. Training step

One training step must do exactly this:

### Step A: generate batch without gradients

```python
model.eval()
with torch.no_grad():
    patch_tokens, _, _ = generate(model, batch_size, cfg)
```

Do not reuse generation log-probs for gradients.

### Step B: re-run masked teacher-forced forward with gradients

```python
model.train()

input_tokens = prepend_bos(patch_tokens)
logits = model(input_tokens)              # [B, N+1, V]
logits = logits[:, :-1, :]                # [B, N, V]

log_probs = log_softmax(logits, dim=-1)
log_q = gather targets patch_tokens
log_q = log_q.sum(dim=1)                  # [B]
```

The model forward must use causal masking.

### Step C: decode lattice and compute energy

```python
spins = tokens_to_lattice(patch_tokens, L, r, c)
energy = ising_energy(spins, J)
```

### Step D: compute surrogate and update

```python
surrogate_loss, Fq = score_function_surrogate(log_q, energy, beta)

optimizer.zero_grad(set_to_none=True)
surrogate_loss.backward()

if grad_clip is not None:
    clip_grad_norm_(model.parameters(), grad_clip)

optimizer.step()
scheduler.step() if enabled
```

Log `Fq`, not the surrogate loss.

## 12. Validation

Validation must generate a fresh batch, possibly larger than the training batch.

```python
with torch.no_grad():
    patch_tokens, log_q_gen, spins = generate(model, val_batch_size, cfg)
```

For accuracy, recompute `log_q` with a full teacher-forced forward and compare occasionally with generated cached `log_q`. Use the teacher-forced value for validation metrics.

### ESS

Use:

\[
\hat w(s_i) = \frac{e^{-\beta E(s_i)}}{q_\theta(s_i)}
= \exp(-\beta E(s_i) - \log q_\theta(s_i)).
\]

\[
\mathrm{ESS}
=
\frac{
\left\langle \hat w \right\rangle^2
}{
\left\langle \hat w^2 \right\rangle
}.
\]

Compute in log-space:

```python
log_w = -beta * energy - log_q

log_mean_w = logsumexp(log_w) - log(B)
log_mean_w2 = logsumexp(2 * log_w) - log(B)

ess = exp(2 * log_mean_w - log_mean_w2)
```

### Free-energy difference

Compute:

```python
Fq = (log_q + beta * energy).mean() / beta
F_exact = exact_ising_free_energy(L, beta, J)
```

Log both:

```python
free_energy_per_spin = Fq / L**2
exact_free_energy_per_spin = F_exact / L**2
free_energy_diff_per_spin = (Fq - F_exact) / L**2
free_energy_relative_diff = (Fq - F_exact) / abs(F_exact)
```

The paper reports relative differences like:

\[
\frac{F_q - F}{|F|}.
\]

Make sure `J` is used consistently in:

1. the exact formula via \(K=\beta J\),
2. the full energy function,
3. AP energy biases,
4. validation free-energy estimates.

A common bug is to double-count bonds in energy or to use `J=1` in the exact formula while using another `J` in training.

## 13. Logging and checkpoints

Use JSONL or CSV logs with records like:

```json
{
  "step": 1000,
  "lr": 0.001,
  "train_free_energy": -34580.1,
  "train_free_energy_per_spin": -2.1101,
  "val_ess": 0.72,
  "val_free_energy_diff_per_spin": 1.2e-5,
  "val_free_energy_relative_diff": 5.6e-6,
  "time_sec": 1234.5
}
```

At every `save_checkpoint_every_n`, save:

```python
{
  "model_config": asdict(model_cfg),
  "train_config": asdict(train_cfg),
  "step": step,
  "model_state_dict": model.state_dict(),
  "optimizer_state_dict": optimizer.state_dict(),
  "scheduler_state_dict": scheduler.state_dict() if exists else None,
  "torch_rng_state": torch.get_rng_state(),
  "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
  "numpy_rng_state": np.random.get_state(),
  "python_rng_state": random.getstate(),
}
```

Support:

```bash
python scripts/train.py --config configs/ising_l120_3x4_ap.yaml --resume checkpoints/step_50000.pt
```

When resuming, restore model, optimizer, scheduler, step counter, and RNG states.

## 14. Optimizer and scheduler

Use AdamW:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_cfg.learning_rate,
    betas=train_cfg.adam_betas,
    weight_decay=train_cfg.weight_decay,
)
```

If cosine scheduling is enabled:

```python
torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=train_cfg.total_steps,
    eta_min=0.0,
)
```

If disabled, use constant LR.

Paper-like defaults:

```yaml
hidden_dim: 128
n_heads: 4
n_blocks: 1        # or 2
use_layernorm: true
init_std: 0.02
learning_rate: 0.001
adam_betas: [0.9, 0.95]
weight_decay: 0.0
batch_size: 4096   # 4096-8192 if memory allows
```

## 15. Example configs

### `configs/ising_l32_2x4.yaml`

```yaml
model:
  L: 32
  patch_r: 2
  patch_c: 4
  hidden_dim: 128
  n_heads: 4
  n_blocks: 1
  use_layernorm: true
  init_std: 0.02
  use_pos_emb: true
  use_ap: false
  beta: 0.44068679350977147
  J: 1.0
  device: cuda
  dtype: float32

train:
  batch_size: 4096
  val_batch_size: 65536
  adam_betas: [0.9, 0.95]
  weight_decay: 0.0
  learning_rate: 0.001
  total_steps: 100000
  use_cosine_scheduler: false
  validate_every_n: 100
  save_logs_every_n: 100
  save_checkpoint_every_n: 5000
  resume_checkpoint: null
  log_dir: runs/ising_l32_2x4
  checkpoint_dir: checkpoints/ising_l32_2x4
  seed: 1234
  grad_clip: null
```

### `configs/ising_l120_3x4_ap.yaml`

```yaml
model:
  L: 120
  patch_r: 3
  patch_c: 4
  hidden_dim: 128
  n_heads: 4
  n_blocks: 1
  use_layernorm: true
  init_std: 0.02
  use_pos_emb: true
  use_ap: true
  beta: 0.44068679350977147
  J: 1.0
  device: cuda
  dtype: float32

train:
  batch_size: 4096
  val_batch_size: 131072
  adam_betas: [0.9, 0.95]
  weight_decay: 0.0
  learning_rate: 0.001
  total_steps: 200000
  use_cosine_scheduler: false
  validate_every_n: 100
  save_logs_every_n: 100
  save_checkpoint_every_n: 5000
  resume_checkpoint: null
  log_dir: runs/ising_l120_3x4_ap
  checkpoint_dir: checkpoints/ising_l120_3x4_ap
  seed: 1234
  grad_clip: null
```

## 16. Required tests

Implement these before long training runs.

### Patch mapping

For several `r,c`, verify:

```python
tokens == patch_to_token(token_to_patch(tokens, r, c))
```

for all tokens.

### Lattice roundtrip

Random spin lattices:

```python
spins == tokens_to_lattice(lattice_to_tokens(spins, r, c), L, r, c)
```

### Energy

For small `L=4`, compare vectorized periodic energy to a brute-force double loop. Confirm every bond is counted once.

### Exact free energy

At beta critical:

```python
F120 / 120**2 ≈ -2.10975198
F128 / 128**2 ≈ -2.10973977
```

### Causal masking

Changing future tokens must not affect earlier logits:

```python
logits_a[:, :k, :] == logits_b[:, :k, :]
```

when sequences only differ after position `k`.

### KV cache

For a fixed token prefix, cached step logits must match the full forward logits at the same position.

### AP

Check:

1. first patch AP uses only internal energy,
2. patch with `pc > 0` includes left boundary,
3. patch with `pr > 0` includes above boundary,
4. periodic wraparound bonds are skipped in AP when the wrapped neighbor is in the future,
5. full energy still includes periodic wraparound bonds.

## 17. Main reproduction runs

The repo should make it easy to reproduce these paper-style comparisons:

1. `L=32`, compare patch shapes such as `1x1`, `2x2`, `2x4`, `4x2`, no AP.
2. `L=120`, compare `2x4` and `3x4`, with and without AP.
3. `L=128`, AP enabled, patch `3x4`, report ESS and relative free-energy difference.
4. Optional large run: `L=180`, AP enabled, patch `3x4`.

The most important outputs are:

```text
training free energy per spin
validation ESS
validation (Fq - F_exact) / |F_exact|
validation (Fq - F_exact) / L^2
```

Use large validation batches for final evaluation. The paper uses up to `1e6` generated samples for final reported quantities; the repo should allow this through `scripts/validate.py`.

## 18. Final design requirements

The implementation must be:

1. **Transparent:** every formula should correspond to a small named function.
2. **Convenient:** one YAML config should fully specify a run.
3. **Efficient:** use FlashAttention through PyTorch SDPA, KV caching for generation, vectorized patch/AP computations, and no Python loops over batch elements.
4. **Reproducible:** support seeds, checkpoints, resuming, and deterministic validation scripts.
5. **Consistent:** one energy convention must be used everywhere. Count every full-lattice nearest-neighbor bond exactly once.