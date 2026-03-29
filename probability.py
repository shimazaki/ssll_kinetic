"""
This code implements the state-space kinetic Ising model described in:
Ken Ishihara, Hideaki Shimazaki. *State-space kinetic Ising model reveals task-dependent entropy flow in sparsely active nonequilibrium neuronal dynamics*. (2025) arXiv:2502.15440

The implementation extends existing libraries available at:
- https://github.com/christiando/ssll_lib.git
- https://github.com/shimazaki/dynamic_corr

Copyright (C) 2025
Authors of the extensions: Ken Ishihara (KenIshihara-17171ken)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import numpy as np

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    @jax.jit
    def _compute_psi_jax(spikes_t, theta_f):
        bias = theta_f[:, 0]
        weights = theta_f[:, 1:]
        logit = bias + spikes_t @ weights.T       # (R, N)
        return jnp.sum(jnp.logaddexp(0, logit), axis=0)  # (N,)

    @jax.jit
    def _compute_psi_batch_jax(spikes, theta_f):
        """Batched PSI over all T timesteps in a single kernel."""
        return jax.vmap(_compute_psi_jax)(spikes, theta_f)  # (T, N)

    @jax.jit
    def _log_marginal_jax(sigma_f, sigma_o, sigma_o_i, theta_f, theta_o,
                          FSUM, spikes_T, V_input, v_input):
        """Full log marginal likelihood on device — slogdet, quadratic, PSI."""
        _, logdet_sigma_f = jnp.linalg.slogdet(sigma_f)
        _, logdet_sigma_o = jnp.linalg.slogdet(sigma_o)

        a = theta_f - theta_o
        Sa = jnp.matmul(sigma_o_i, a[..., jnp.newaxis]).squeeze(-1)
        b = 0.5 * jnp.sum(a * Sa, axis=-1)

        # Observation input offset: (T, N)
        obs_offset = jnp.matmul(v_input, V_input.T)  # (T, N)

        bias = theta_f[:, :, 0][:, jnp.newaxis, :]
        weights = theta_f[:, :, 1:]
        logit = bias + jnp.matmul(spikes_T, jnp.swapaxes(weights, -2, -1))
        logit = logit + obs_offset[:, jnp.newaxis, :]  # add offset
        PSI = jnp.sum(jnp.logaddexp(0, logit), axis=1)

        q = jnp.sum(theta_f * FSUM, axis=-1) + obs_offset * FSUM[:, :, 0] - PSI - b
        return jnp.sum(0.5 * logdet_sigma_f - 0.5 * logdet_sigma_o + q)

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

def log_marginal(emd):
    """
    Computes the log marginal likelihood of the data given the current state estimates.

    :param container.EMData emd:
        An EMData-like object containing:
          - sigma_f (numpy.ndarray): Filtered covariance, shape (T, N, N+1, N+1).
          - sigma_o (numpy.ndarray): One-step predicted covariance, shape (T, N, N+1, N+1).
          - sigma_o_i (numpy.ndarray): Inverses of sigma_o, shape (T, N, N+1, N+1).
          - theta_f (numpy.ndarray): Filtered parameter estimates, shape (T, N, N+1).
          - theta_o (numpy.ndarray): Predicted parameter estimates, shape (T, N, N+1).
          - FSUM (numpy.ndarray): Precomputed feature sums, shape (T, N, N+1).
          - spikes (numpy.ndarray): Spike data, shape (T, R, N).
          - T (int): Number of time bins.
          - R (int): Number of trials.
    :returns:
        float
            The computed log marginal likelihood of the data given the current estimates.
    """
    if _HAS_JAX:
        if emd.V is not None:
            V_jax = jnp.asarray(emd.V)
            v_jax = jnp.asarray(emd.v[:emd.T])
        else:
            V_jax = jnp.zeros((emd.N, 1))
            v_jax = jnp.zeros((emd.T, 1))
        return float(_log_marginal_jax(
            jnp.asarray(emd.sigma_f),
            jnp.asarray(emd.sigma_o),
            jnp.asarray(emd.sigma_o_i),
            jnp.asarray(emd.theta_f),
            jnp.asarray(emd.theta_o),
            jnp.asarray(emd.FSUM),
            jnp.asarray(emd.spikes[:emd.T]),
            V_jax,
            v_jax,
        ))

    # 1) Compute log-determinants of sigma_f and sigma_o
    _, logdet_sigma_f = np.linalg.slogdet(emd.sigma_f)
    _, logdet_sigma_o = np.linalg.slogdet(emd.sigma_o)

    # 2) Quadratic penalty term: b[t,i] = 0.5 * a[t,i] @ sigma_o_i[t,i] @ a[t,i]
    a = emd.theta_f - emd.theta_o
    Sa = np.matmul(emd.sigma_o_i, a[..., np.newaxis]).squeeze(-1)  # (T,N,N+1)
    b = 0.5 * np.sum(a * Sa, axis=-1)                              # (T,N)

    # 3-4) PSI = sum_r logaddexp(0, theta_f @ F1_r) — avoid allocating (T,R,N+1) F_1
    #   theta_f @ F1.T = theta_f[:,n,0] (bias) + spikes[:T] @ theta_f[:,n,1:].T (couplings)
    bias = emd.theta_f[:, :, 0][:, np.newaxis, :]           # (T, 1, N)
    weights = emd.theta_f[:, :, 1:]                          # (T, N, N)
    logit = bias + np.matmul(emd.spikes[:emd.T], weights.swapaxes(-2, -1))  # (T, R, N)

    # Add observation input offset
    if emd.V is not None:
        obs_offset = emd.v[:emd.T] @ emd.V.T  # (T, N)
        logit = logit + obs_offset[:, np.newaxis, :]
    else:
        obs_offset = np.zeros((emd.T, emd.N))

    PSI = np.sum(np.logaddexp(0, logit), axis=1)             # (T, N)

    # 5) Compute q = <theta_f, FSUM> + offset*spike_count - PSI - b
    q = np.sum(emd.theta_f * emd.FSUM, axis=-1) + obs_offset * emd.FSUM[:, :, 0] - PSI - b

    # 6) Combine determinant and q terms for final log marginal likelihood
    log_p = np.sum(0.5 * logdet_sigma_f - 0.5 * logdet_sigma_o + q)
    return log_p
