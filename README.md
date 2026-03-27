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
# Log marginal likelihood: emd.mllk
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

## Entropy flow

```python
sf_bath, sr_bath, s_bath, M = entropy_flow.compute_entropy_flow(emd)
# sf_bath: forward conditional entropy, shape (T, N)
# sr_bath: reverse conditional entropy, shape (T, N)
# s_bath:  net entropy flow (dissipative), shape (T, N)
# M:       mean-field spike probabilities, shape (T, N)
```

For stationary models (`emd.T == 1` after pooling with `stationary=True`), `compute_entropy_flow` automatically iterates the mean-field equation m = f(theta, m) from the empirical spike mean to the fixed point m\*, then computes entropy flow at (theta, m\*, m\*). The returned arrays have shape `(1, N)`.

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

## Running tests

```bash
cd /path/to/ssll_kinetic
python -m unittest testing -v
```

## License

GNU General Public License v3.0
