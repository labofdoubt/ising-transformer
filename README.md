# Ising Transformer

From-scratch PyTorch implementation of a transformer-based autoregressive sampler for the 2D Ising model, following the recipe from *Sampling two-dimensional spin systems with transformers*:

- Paper: https://arxiv.org/abs/2604.27738v1

The code does not depend on external transformer repos. Core pieces such as patch tokenization, causal self-attention, KV-cached generation, approximate-probability energy bias, variational free-energy training, and exact finite-size Ising validation are implemented directly in this repository.

## Overview

The sampler is a decoder-only autoregressive transformer over lattice patches rather than individual spins.

The workflow is:

1. Split an `L x L` Ising lattice into non-overlapping `patch_r x patch_c` subgrids.
2. Map each patch to an integer token.
3. Generate patch tokens autoregressively with a transformer.
4. Decode generated tokens back to spin lattices.
5. Minimize the empirical variational free energy of the generated batch.

Validation is based on two main quantities:

- exact finite-`L` free energy of the periodic 2D Ising model
- effective sample size (ESS)

The repository also supports AP, the approximate-probability energy bias used in the paper. AP adds a local physical bias to the logits based on internal patch bonds and already generated neighboring patches.

## Repository Layout

```text
tvan/
  config.py
  patches.py
  lattice.py
  physics.py
  exact_ising.py
  model.py
  ap.py
  generation.py
  losses.py
  checkpoint.py
  logging_utils.py
scripts/
  train.py
  validate.py
  sample.py
configs/
  ising_l32_2x4.yaml
  ising_l120_3x4_ap.yaml
notebooks/
  colab_training_and_inference.ipynb
tests/
  ...
```

## Colab Entry Point

The main entry point for interactive experimentation is [notebooks/colab_training_and_inference.ipynb](/Users/sergeyalekseev/Desktop/ML_projects/ising_transformer/notebooks/colab_training_and_inference.ipynb).

The notebook is set up to:

- clone the repository in Colab
- mount Google Drive
- create a YAML config for a run
- launch training
- read saved logs and plot validation ESS and relative error versus training step
- load a checkpoint and generate samples for downstream observables such as magnetization and susceptibility

The notebook keeps the repo itself under Colab local storage and uses Google Drive paths for persistent logs and checkpoints.

## Configs

Runs are configured through a single YAML file with two sections:

- `model`
- `train`

`model` controls the lattice and transformer:

- `L`: lattice size for an `L x L` system
- `patch_r`, `patch_c`: patch height and width
- `hidden_dim`, `n_heads`, `n_blocks`: transformer size
- `use_layernorm`, `use_pos_emb`
- `use_ap`: whether to enable the approximate-probability bias
- `beta`, `J`
- `device`, `dtype`

`train` controls optimization and output:

- `batch_size`, `val_batch_size`
- `learning_rate`, `adam_betas`, `weight_decay`
- `total_steps`
- `use_cosine_scheduler`
- `validate_every_n`
- `save_logs_every_n`
- `save_checkpoint_every_n`
- `resume_checkpoint`
- `log_dir`, `checkpoint_dir`
- `seed`, `grad_clip`

Example configs are provided in:

- [configs/ising_l32_2x4.yaml](/Users/sergeyalekseev/Desktop/ML_projects/ising_transformer/configs/ising_l32_2x4.yaml)
- [configs/ising_l120_3x4_ap.yaml](/Users/sergeyalekseev/Desktop/ML_projects/ising_transformer/configs/ising_l120_3x4_ap.yaml)

## Local Training

Run training with:

```bash
python scripts/train.py --config configs/ising_l32_2x4.yaml
```

Resume from a checkpoint with:

```bash
python scripts/train.py \
  --config configs/ising_l32_2x4.yaml \
  --resume checkpoints/ising_l32_2x4/step_5000.pt
```

At startup, the training script prints the exact free energy per spin for the configured lattice. During training it reports:

- training free energy per spin
- training relative free-energy error
- validation ESS
- validation free-energy per-spin absolute error
- validation relative free-energy error

Metrics are written to:

- `metrics.jsonl`
- `metrics.csv`

Checkpoints include:

- model state
- optimizer state
- scheduler state
- RNG states
- serialized model and training configs

## Other Scripts

Validate a trained model:

```bash
python scripts/validate.py \
  --config configs/ising_l32_2x4.yaml \
  --checkpoint checkpoints/ising_l32_2x4/step_5000.pt \
  --num-samples 100000
```

This generates fresh samples, recomputes teacher-forced log-probabilities, and reports free-energy metrics and ESS.

Sample and save generated lattices:

```bash
python scripts/sample.py \
  --config configs/ising_l32_2x4.yaml \
  --checkpoint checkpoints/ising_l32_2x4/step_5000.pt \
  --batch-size 128 \
  --output samples_run.npz
```

## Testing

Run the test suite with:

```bash
pytest -q
```

The tests cover:

- patch/token roundtrips
- lattice roundtrips
- Ising energy conventions
- exact finite-size free energy regression
- causal masking
- KV-cache consistency
- AP boundary handling
