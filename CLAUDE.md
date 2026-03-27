# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

State-Space Kinetic Ising Model for neural spike train analysis. Implements an EM algorithm to infer time-varying parameters from binary spike data using a logistic state-space framework. Based on Ishihara & Shimazaki (2025), arXiv:2502.15440.

## Running Tests

```bash
# From within the package directory
cd /mnt/disk1/home/hideaki/Dropbox/lab/github/ssll_kinetic
python -m unittest testing -v
```

Uses `unittest` (no pytest config).

## Dependencies

numpy, matplotlib, IPython, joblib, numba. Optional: jax[cuda12] for GPU acceleration (requires Python >=3.11 and CUDA 12; use `py311` conda environment on otto/diesel). Falls back to numpy automatically when JAX is unavailable.

**GPU vs CPU guidance**: GPU (~15x) for stationary mode and non-stationary with N>=20. For non-stationary with N<=10, numpy (CPU) is faster — per-timestep matrices are too small for GPU overhead. N (neurons) is the dominant scaling factor for GPU benefit due to O((N+1)^3) NR solves.

## Architecture

Single-level Python package (`ssll_kinetic/`) with no subpackages:

- **`__init__.py`** — `run(spikes, max_iter=100, mstep=True, state_cov=0.5, stationary=False)` entry point: orchestrates the EM loop (E-step -> M-step -> likelihood -> convergence check). Computes AIC after convergence. With `stationary=True`, pools all T×R transitions into a single time step for time-independent estimation.
- **`container.py`** — `EMData(spikes)` class: holds all algorithm state. Takes only spikes; internally initializes state_cov (0.5*I), init_cov (I), init_theta (zeros), F (identity). Pre-computes `FSUM` sufficient statistics.
- **`exp_max.py`** — Core inference with optional JAX GPU acceleration:
  - `e_step_filter` (forward pass, Newton-Raphson MAP) — with JAX, the entire T-loop runs on-device via `jax.lax.scan` wrapping `jax.lax.while_loop` (zero host↔device round-trips); uses init carry trick (`init_cov - state_cov`) to avoid t==0 branching
  - `e_step_smooth` (backward pass) — with JAX, runs via `jax.lax.scan(reverse=True)` on-device
  - `compute_eta_G` — sigmoid + Fisher information; dispatches to `_compute_eta_G_jax` (JIT) when JAX available, numpy fallback otherwise
  - `e_step_filter_parallel` / `e_step_smooth_parallel` (joblib variants, commented out by default)
  - `@njit` compiled helpers for parallel filtering (`compute_eta_G_parallel`, `process_single_i`)
  - M-step: `get_diagonal_Q` (default), `get_full_Q` (full dense), `get_scalar_Q` (isotropic) + `get_init_cov` (init_cov update)
- **`probability.py`** — `log_marginal()`: with JAX, entire computation (slogdet, quadratic, PSI) runs in a single JIT kernel via `_log_marginal_jax`; numpy fallback uses vectorized `np.linalg.slogdet` and `np.matmul`
- **`synthesis.py`** — Data generation: `generate_thetas`, `generate_thetas_fixed_seed`, `generate_spikes` (spike sampling), `shuffle_spikes`
- **`entropy_flow.py`** — Entropy flow computation from estimated parameters (theta_s):
  - `compute_entropy_flow(emd)` — main entry: returns forward/reverse/net entropy flow time series + mean-field spike probabilities
  - `compute_dissipation(a, m, m_p)` — per-time-step forward/reverse/net entropy flow per neuron
  - `compute_mean_field(a, m_p)` — mean-field spike probability update
  - Two equivalent formulations: chi-based (`update_S`, `update_S_re`) and h-psi decomposition (`update_S_alt`, `update_S_re_alt`); see Eqs. 47-48, 57-58 in the paper
- **`testing.py`** — 16 unit tests covering EM, entropy flow, parameter recovery, formulation equivalence, stationary analysis, JAX/numpy parity, and JAX scan/numpy parity

## Data Dimensions

- **Spikes**: `(T+1, R, N)` — T time bins, R trials, N neurons
- **State covariance / init_cov**: `(N, N+1, N+1)` — per-neuron, includes bias term
- **Theta parameters**: `(N, N+1)` — bias + N lagged spike dependencies per neuron

## Key Constants

- `GA_CONVERGENCE = 1e-4` — Newton-Raphson tolerance in filtering (exp_max.py)
- `MAX_GA_ITERATIONS = 5000` — Newton-Raphson iteration cap (exp_max.py)
- `CONVERGED = 1e-5` — EM convergence threshold (container.py)

## Import Pattern

Uses absolute imports (`from ssll_kinetic.probability import *`). Import as a package from the parent directory.

## Upstream

Synced from https://github.com/KenIshihara-17171ken/Non_equ/tree/master/ssll_kinetic
