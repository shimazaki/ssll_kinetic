# ssll_kinetic

State-space kinetic Ising model for neural spike train analysis. Implements an EM algorithm to infer time-varying field and coupling parameters from binary spike data using a logistic state-space framework.

Based on: Ken Ishihara, Hideaki Shimazaki. *State-space kinetic Ising model reveals task-dependent entropy flow in sparsely active nonequilibrium neuronal dynamics*. (2025) [arXiv:2502.15440](https://arxiv.org/abs/2502.15440)

Upstream repository: [KenIshihara-17171ken/Non_equ](https://github.com/KenIshihara-17171ken/Non_equ/tree/master/ssll_kinetic)

## Dependencies

numpy, matplotlib, joblib, numba. Optional: jax[cuda12] for GPU acceleration (falls back to numpy automatically).

Install via conda (using the `ssll` environment) or pip.

## Quick start

```python
import numpy as np
import ssll_kinetic
from ssll_kinetic import synthesis, entropy_flow

# Generate synthetic spike data
T, R, N = 500, 200, 2  # time bins, trials, neurons
np.random.seed(42)
THETA = synthesis.generate_thetas(T, N, mu=-2.0, sigma=50.0, alpha=12.0)
np.random.seed(1)
spikes = synthesis.generate_spikes(T, R, N, THETA)

# Run the EM algorithm
emd = ssll_kinetic.run(spikes, max_iter=100)

# Estimated parameters: emd.theta_s (shape: T, N, N+1)
# Log marginal likelihood: emd.mll
```

## The `state_cov` parameter

The `state_cov` parameter controls how the state covariance matrix Q is estimated in the M-step. Q is estimated **per neuron**, each with shape `(N+1, N+1)`.

| `state_cov` value | Initial Q | M-step update | Use case |
|---|---|---|---|
| scalar (e.g. `0.5`, default) | `0.5 * I` | `get_scalar_Q` (isotropic) | Fewest parameters |
| vector shape `(N+1,)` | `diag(v)` | `get_diagonal_Q` | Per-parameter variance |
| matrix shape `(N+1, N+1)` | copy of matrix | `get_full_Q` (full dense) | Cross-parameter covariance |
| `0` or `None` | zeros | No Q update | Fixed dynamics |

```python
# Scalar Q (isotropic, default)
emd = ssll_kinetic.run(spikes, max_iter=100, state_cov=0.5)

# Diagonal Q
emd = ssll_kinetic.run(spikes, max_iter=100, state_cov=0.5*np.ones(N+1))

# Full Q
emd = ssll_kinetic.run(spikes, max_iter=100, state_cov=0.5*np.identity(N+1))

# No M-step (fixed Q)
emd = ssll_kinetic.run(spikes, max_iter=100, mstep=False)
```

## Stationary analysis

Pass `stationary=True` to fit a time-independent kinetic Ising model by pooling all T×R transition observations into a single time step. This is the kinetic analogue of ssll's stationary analysis.

```python
# Fit a single set of parameters across all time steps
emd = ssll_kinetic.run(spikes, max_iter=100, stationary=True)

# Single theta estimate: shape (N, N+1)
theta = emd.theta_s[0]

# AIC with k = N*(N+1) free parameters
print(emd.aic)
```

Internally, spikes `(T+1, R, N)` are reshaped to `(2, T*R, N)` and fitted with `state_cov=0` (Q fixed at zero), so the EM estimates a single pooled theta.

**Observation input with stationary pooling**: When `stationary=True`, observation input `v` is preserved through pooling: it is reshaped from `(T, R, d_v)` to `(1, T*R, d_v)` so each pooled trial keeps its own observation input vector. This makes V identifiable under `stationary=True`.

**Limitation with state input**: State input `u` is time-averaged when `stationary=True`, and U is not identifiable with T=1 (the state transition is eliminated by pooling). To fit a stationary model with time-varying state input, use `state_cov=0` instead.

```python
# Stationary model with observation input (v preserved)
emd = ssll_kinetic.run(spikes, stationary=True, v=v)

# Alternative: state_cov=0 keeps all T time steps
emd = ssll_kinetic.run(spikes, state_cov=0, v=v)
```

## Exogenous input

The model supports two types of exogenous input:

**State input** `u` / `U`: Drives the state equation θ\_t = θ\_{t-1} + U·u\_t + ξ\_t. The gain matrix U is learned via a closed-form M-step.

```python
u = np.random.randn(T, d_u)  # (T, d_u) state input signal
emd = ssll_kinetic.run(spikes, u=u)
# emd.U: (N, N+1, d_u) learned state input gain
```

**Observation input** `v` / `V`: Directly offsets the firing probability in the observation model: r\_i = σ(θ\_{t,i}' f(x\_{t-1}) + V\_i' v\_t). V is learned via Newton-Raphson in the M-step.

```python
v = np.random.randn(T, d_v)  # (T, d_v) observation input signal
emd = ssll_kinetic.run(spikes, v=v)
# emd.V: (N, d_v) learned observation input gain
```

**Combined**: Both inputs can be used simultaneously.

```python
emd = ssll_kinetic.run(spikes, u=u, v=v)
```

**Shapes**: `u` is `(T, d_u)`, `v` is `(T, d_v)`. Results: `emd.U` is `(N, N+1, d_u)`, `emd.V` is `(N, d_v)`.

**Identifiability note**: The observation input V·v\_t and the bias component of θ\_{t,i,0} can trade off. With sufficient temporal variation in v\_t and enough trials, both are identifiable. When v\_t = 0, the model reduces to the standard form.

## Entropy flow

```python
sf_bath, sr_bath, s_bath, M = entropy_flow.compute_entropy_flow(emd)
# sf_bath: forward conditional entropy, shape (T, N)
# sr_bath: reverse conditional entropy, shape (T, N)
# s_bath:  net entropy flow (dissipative), shape (T, N)
# M:       mean-field spike probabilities, shape (T, N)
```

**Per-trial entropy flow**: When observation input `v` is trial-specific, each trial experiences a different external drive, producing distinct mean-field trajectories and entropy flows. Use `compute_entropy_flow_per_trial` to get per-trial results:

```python
sf, sr, s_net, M = entropy_flow.compute_entropy_flow_per_trial(emd)
# All arrays have shape (T, R, N) — one trajectory per trial
# Trial-averaged result: sf.mean(axis=1) matches compute_entropy_flow output
```

When `V` is None (no observation input), all trials are identical and `compute_entropy_flow_per_trial` broadcasts the single result to `(T, R, N)`. `compute_entropy_flow` calls the per-trial version internally and averages over R.

For stationary models (`emd.T == 1` after pooling with `stationary=True`), both functions automatically iterate the mean-field equation m = f(theta, m) from the empirical spike mean to the fixed point m\*, then compute entropy flow at (theta, m\*, m\*). The returned arrays have shape `(1, N)` or `(1, R, N)`.

```python
emd = ssll_kinetic.run(spikes, max_iter=100, stationary=True)
sf, sr, s_net, M = entropy_flow.compute_entropy_flow(emd)
m_star = M[0]  # fixed-point spike probabilities, shape (N,)
```

## Performance

Core routines are vectorized with NumPy broadcasting and `einsum`, avoiding Python-level loops over trials/time/neurons in the hot path:

- **FSUM** computation uses a single `einsum` over `(T, R, N)` arrays instead of triple loops
- **Newton-Raphson filtering** builds the feature matrix once per time step and uses BLAS matrix multiplies for eta/G computation, with peak memory `O(R * N)` instead of `O(R * N^2)`
- **Q estimation** exploits the algebraic identity `outer(a,a) - outer(a,b) - outer(b,a) + outer(b,b) = outer(a-b, a-b)` to contract directly to `(N, N+1, N+1)`

### JAX GPU acceleration (optional)

When JAX is available, the entire E-step runs on-device with zero host round-trips per EM iteration:

- **Forward filter** (`e_step_filter`): `jax.lax.scan` over T timesteps, each containing a `jax.lax.while_loop` Newton-Raphson solve — compiled into a single XLA program
- **Backward smoother** (`e_step_smooth`): `jax.lax.scan(reverse=True)` over T-1 timesteps
- **Log marginal likelihood** (`log_marginal`): slogdet, quadratic penalty, and PSI in one JIT kernel

Falls back to numpy automatically when JAX is unavailable.

**Benchmark: per-iteration time** (T=500, R=200; CPU = Xeon 6258R on carnot, GPU = A100-80GB on otto):

Non-stationary:

| N | CPU (numpy) | GPU (JAX) | Speedup |
|---|---|---|---|
| 5 | 0.17s | 0.20s | 0.8x |
| 10 | 0.32s | 0.28s | 1.2x |
| 20 | 0.94s | 0.46s | 2.0x |
| 50 | 15.9s | 2.7s | 5.8x |
| 100 | 70.2s | 19.9s | 3.5x |

Stationary (`stationary=True`):

| N | CPU (numpy) | GPU (JAX) | Speedup |
|---|---|---|---|
| 5 | 0.11s | 0.007s | 15x |
| 10 | 0.07s | 0.004s | 18x |
| 20 | 0.31s | 0.010s | 31x |
| 50 | 4.3s | 0.040s | 107x |
| 100 | 16.0s | 0.13s | 121x |

**When to use GPU**: Always for stationary mode (15-121x speedup). For non-stationary, GPU helps at N >= 20 (2-6x speedup); at N <= 10, CPU is comparable or faster. N (neurons) is the dominant scaling factor because the state covariance and Fisher information matrices grow as (N+1)x(N+1), and the NR linear solve is O((N+1)^3) per neuron per iteration.

To force the numpy path when JAX is installed, set `exp_max._HAS_JAX = False` and `probability._HAS_JAX = False` before calling `run()`.

## Important: spike array dtype

The `spikes` array must use a dtype that does not overflow during internal `einsum` accumulation (e.g., `float64`, `float32`, or `int32`). **Do not pass `int8` arrays** — `container.py` computes `FSUM` via `np.einsum('trn,trm->tnm', current, prev)`, which accumulates in the input dtype. With `int8`, sums exceeding 127 silently overflow to negative values, corrupting the sufficient statistics and causing Newton-Raphson divergence.

```python
# Wrong — int8 overflows for R > 127
spikes = binary_data.astype(np.int8)

# Correct — float64 is safest (matches all internal computations)
spikes = binary_data.astype(np.float64)
```

## Running tests

```bash
cd /path/to/ssll_kinetic
python -m unittest testing -v
```

## License

GNU General Public License v3.0
