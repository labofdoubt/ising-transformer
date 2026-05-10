# Ising Transformer

From-scratch PyTorch implementation of a transformer-based autoregressive sampler for the 2D Ising model, following the recipe from *Białas et al., [Sampling two-dimensional spin systems with transformers](https://arxiv.org/abs/2604.27738v1)*:

The code does not depend on external transformer repos.

## Overview

This repository implements an autoregressive transformer sampler for the two-dimensional Ising model.

### Lattice Representation

- The lattice has size `L x L`.
- The lattice is split into non-overlapping patches of size `r x c`.
- Each patch is flattened and mapped to a token ID.
- The full lattice is represented as a sequence of tokens.

### Sample Generation

The transformer generates lattices autoregressively: it samples one patch at a time, and each patch is conditioned on the patches generated before it.

### Training Objective

Training minimizes the empirical variational free energy of a batch of lattices. For each sampled lattice, this uses:

- the model log-probability of the generated lattice,
- the Ising energy of the lattice (periodic boundary conditions are used).

### Validation

Validation uses two metrics:

- **Free-energy error:** compares the model’s estimated free energy with the exact finite-`L` Ising free energy.
- **ESS:** effective sample size, following the definition used in the [paper](https://arxiv.org/abs/2604.27738v1).

The repository also supports AP (approximate probability), the energy bias introduced in the paper. AP adds a local energy-based bias to the logits, using interactions within the current patch and with already generated neighboring patches.

## Colab Entry Point

The main entry point for the repo is [notebooks/colab_training_and_inference.ipynb](/Users/sergeyalekseev/Desktop/ML_projects/ising_transformer/notebooks/colab_training_and_inference.ipynb).

The notebook shows how to:

- clone the repository in Colab
- create a config for a run
- launch training
- read saved logs and plot validation ESS and relative error vs. training step
- load a checkpoint and estimate observables such as magnetization and susceptibility from generated samples

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