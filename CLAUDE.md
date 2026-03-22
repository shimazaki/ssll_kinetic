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

numpy, matplotlib, IPython, joblib, numba. Use the `ssll` conda environment.

## Architecture

Single-level Python package (`ssll_kinetic/`) with no subpackages:

- **`__init__.py`** — `run(spikes, max_iter=100, mstep=True)` entry point: orchestrates the EM loop (E-step -> M-step -> likelihood -> convergence check). Computes AIC after convergence.
- **`container.py`** — `EMData(spikes)` class: holds all algorithm state. Takes only spikes; internally initializes state_cov (0.5*I), init_cov (I), init_theta (zeros), F (identity). Pre-computes `FSUM` sufficient statistics.
- **`exp_max.py`** — Core inference using `np.einsum` for vectorized operations:
  - `filter_function` (forward pass, Newton-Raphson MAP)
  - `smoothing_function` (backward pass)
  - `filter_function_Parallel` / `smoothing_function_Parallel` (joblib variants, commented out by default)
  - `@njit` compiled helpers for parallel filtering (`cal_eta_G_Para`, `process_single_i`)
  - M-step: `get_diagonal_Q` (default), `get_Q` (full dense), `get_scalar_q` (isotropic) + `get_Sigma` (init_cov update)
- **`probability.py`** — `log_marginal()`: vectorized log marginal likelihood using `np.linalg.slogdet` and `np.einsum`
- **`synthesis.py`** — Data generation: `get_THETA_gaussian_process`, `get_THETA_gaussian_process_fixed_seed_per_neuron`, `get_S_function` (spike sampling), `shuffle_spikes`
- **`macro.py`** — Entropy flow computation from estimated parameters (theta_s):
  - `calculate_entropy_flow(emd)` — main entry: returns forward/reverse/net entropy flow time series + mean-field spike probabilities
  - `Dissipation_en(a, m, m_p)` — per-time-step forward/reverse/net entropy flow per neuron
  - `computation_m(a, m_p)` — mean-field spike probability update
  - Two equivalent formulations: chi-based (`update_S`, `update_S_re`) and h-psi decomposition (`update_S_alt`, `update_S_re_alt`); see Eqs. 47-48, 57-58 in the paper
- **`testing.py`** — 11 unit tests covering EM, entropy flow, parameter recovery, and formulation equivalence

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
