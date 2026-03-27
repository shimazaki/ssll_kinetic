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
from joblib import Parallel, delayed
from numba import njit

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    @jax.jit
    def _compute_eta_G_jax(theta, F1):
        r = jax.nn.sigmoid(F1 @ theta.T)            # (R, N)
        eta = r.T @ F1                               # (N, N+1)
        w = r * (1 - r)                              # (R, N)
        G = jnp.einsum('rn,ri,rj->nij', w, F1, F1)  # (N, N+1, N+1)
        return eta, G

    def _nr_loop_jax(theta_f, F1, FSUM, sigma_o_i, theta_o, R,
                     ga_convergence, max_ga_iterations):
        """Full Newton-Raphson loop on device via jax.lax.while_loop."""
        N, Np1 = theta_f.shape

        def _cond(state):
            theta, sigma_f, max_dlpo, iterations = state
            return (max_dlpo > ga_convergence) & (iterations < max_ga_iterations)

        def _body(state):
            theta, _, _, iterations = state
            eta, G = _compute_eta_G_jax(theta, F1)
            diff = theta - theta_o
            dlpo = eta - FSUM + jnp.matmul(
                sigma_o_i, diff[..., jnp.newaxis]
            ).squeeze(-1)
            ddlpo = G + sigma_o_i
            ddlpo_i = jnp.linalg.inv(ddlpo)
            theta = theta - jnp.matmul(
                ddlpo_i, dlpo[..., jnp.newaxis]
            ).squeeze(-1)
            max_dlpo = jnp.amax(jnp.absolute(dlpo)) / R
            return theta, ddlpo_i, max_dlpo, iterations + 1

        init_sigma = jnp.zeros((N, Np1, Np1))
        init_state = (theta_f, init_sigma, jnp.inf, 0)
        theta_final, sigma_f, _, iterations = jax.lax.while_loop(
            _cond, _body, init_state)
        return theta_final, sigma_f, iterations

    _nr_loop_jax = jax.jit(_nr_loop_jax, static_argnums=(5,))

    def _e_step_filter_jax_fn(spikes_T, FSUM, init_theta, init_cov, state_cov,
                               R, ga_convergence, max_ga_iterations):
        """Full forward filter via jax.lax.scan — zero host round-trips."""
        N = init_theta.shape[0]
        Np1 = init_theta.shape[1]

        ones_col = jnp.ones((spikes_T.shape[0], spikes_T.shape[1], 1))
        F1_all = jnp.concatenate([ones_col, spikes_T], axis=2)  # (T, R, N+1)

        # Init carry trick: sigma_f_prev = init_cov - state_cov so that
        # sigma_o = carry_sigma + state_cov = init_cov at t=0.
        init_carry = (init_theta, init_cov - state_cov)

        def scan_body(carry, xs):
            theta_f_prev, sigma_f_prev = carry
            F1_t, FSUM_t = xs

            theta_o = theta_f_prev
            sigma_o = sigma_f_prev + state_cov
            sigma_o_i = jnp.linalg.inv(sigma_o)

            # Newton-Raphson via while_loop
            def _cond(state):
                _, _, max_dlpo, iterations = state
                return (max_dlpo > ga_convergence) & (iterations < max_ga_iterations)

            def _body(state):
                theta, _, _, iterations = state
                r = jax.nn.sigmoid(F1_t @ theta.T)
                eta = r.T @ F1_t
                w = r * (1 - r)
                G = jnp.einsum('rn,ri,rj->nij', w, F1_t, F1_t)

                diff = theta - theta_o
                dlpo = eta - FSUM_t + jnp.matmul(
                    sigma_o_i, diff[..., jnp.newaxis]).squeeze(-1)
                ddlpo = G + sigma_o_i
                ddlpo_i = jnp.linalg.inv(ddlpo)
                theta = theta - jnp.matmul(
                    ddlpo_i, dlpo[..., jnp.newaxis]).squeeze(-1)
                max_dlpo = jnp.amax(jnp.absolute(dlpo)) / R
                return theta, ddlpo_i, max_dlpo, iterations + 1

            init_sigma = jnp.zeros((N, Np1, Np1))
            init_state = (theta_o, init_sigma, jnp.inf, 0)
            theta_f, sigma_f, _, iters = jax.lax.while_loop(
                _cond, _body, init_state)

            return (theta_f, sigma_f), (theta_f, sigma_f, theta_o, sigma_o,
                                        sigma_o_i, iters)

        _, outputs = jax.lax.scan(scan_body, init_carry, (F1_all, FSUM))
        return outputs

    _e_step_filter_jax = jax.jit(
        _e_step_filter_jax_fn, static_argnums=(5, 6, 7))

    @jax.jit
    def _e_step_smooth_jax(theta_f, sigma_f, theta_o, sigma_o, sigma_o_i):
        """Full backward smoother via jax.lax.scan(reverse=True)."""
        T = theta_f.shape[0]

        init_carry = (theta_f[T - 1], sigma_f[T - 1])

        scan_inputs = (
            sigma_f[:T - 1],
            sigma_o_i[1:T],
            theta_f[:T - 1],
            theta_o[1:T],
            sigma_o[1:T],
        )

        def scan_body(carry, xs):
            theta_s_next, sigma_s_next = carry
            sigma_f_t, sigma_o_i_tp1, theta_f_t, theta_o_tp1, sigma_o_tp1 = xs

            A_t = jnp.matmul(sigma_f_t, sigma_o_i_tp1)

            diff_theta = theta_s_next - theta_o_tp1
            theta_s_t = theta_f_t + jnp.matmul(
                A_t, diff_theta[..., jnp.newaxis]).squeeze(-1)

            diff_sigma = sigma_s_next - sigma_o_tp1
            sigma_s_t = sigma_f_t + jnp.matmul(
                jnp.matmul(A_t, diff_sigma),
                jnp.swapaxes(A_t, -2, -1))

            lag_one_cov_t = jnp.matmul(A_t, sigma_s_next)

            return (theta_s_t, sigma_s_t), (theta_s_t, sigma_s_t, A_t,
                                            lag_one_cov_t)

        _, outputs = jax.lax.scan(
            scan_body, init_carry, scan_inputs, reverse=True)
        theta_s_scan, sigma_s_scan, A_scan, lag_one_cov_scan = outputs

        theta_s = jnp.concatenate([theta_s_scan, theta_f[T - 1:T]], axis=0)
        sigma_s = jnp.concatenate([sigma_s_scan, sigma_f[T - 1:T]], axis=0)

        return theta_s, sigma_s, A_scan, lag_one_cov_scan

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

MAX_GA_ITERATIONS = 5000
GA_CONVERGENCE = 1e-4

def e_step(emd):
    """
    Performs the E-step of the EM algorithm, combining a filtering and smoothing procedure.

    :param emd: container.EMData
        The data structure holding all necessary model parameters and spike data.
    :returns: None
    """
    e_step_filter(emd)
    # If parallel filtering is preferred, uncomment:
    #e_step_filter_parallel(emd)

    e_step_smooth(emd)
    # If parallel smoothing is preferred, uncomment:
    # e_step_smooth_parallel(emd)

def m_step(emd):
    """
    Performs the M-step of the EM algorithm, updating state covariance and initial covariance matrices.
    The Q estimation method is selected by emd.state_cov_0 (set via the state_cov parameter):
      - scalar → get_scalar_Q (isotropic)
      - vector (N+1,) → get_diagonal_Q (diagonal)
      - matrix (N+1, N+1) → get_full_Q (full dense)

    :param emd: container.EMData
        The data structure holding the current state of model parameters.
    :returns: None
    """
    sc0 = emd.state_cov_0
    if sc0 is None or (np.isscalar(sc0) and sc0 == 0):
        pass  # Stationary: keep Q fixed at zero
    elif np.isscalar(sc0):
        get_scalar_Q(emd)
    else:
        sc0 = np.asarray(sc0)
        if sc0.ndim == 1:
            get_diagonal_Q(emd)
        elif sc0.ndim == 2:
            get_full_Q(emd)
    get_init_cov(emd)

def get_init_theta(emd):
    """
    Updates the initial theta estimate to the smoothed estimate at time 0.

    :param emd: container.EMData
        The data structure holding the smoothed parameter estimates.
    :returns: numpy.ndarray
        The updated initial theta of shape (N, N+1).
    """
    emd.init_theta = np.zeros((emd.N, emd.N+1))
    for i in range(emd.N):
        emd.init_theta[i] = emd.theta_s[0, i]
    return emd.init_theta

def get_init_cov(emd):
    """
    Updates the initial covariance matrices based on the smoothed estimates at time 0.

    :param emd: container.EMData
        The data structure containing smoothed parameters and covariances.
    :returns: numpy.ndarray
        The updated init_cov of shape (N, N+1, N+1).
    """
    emd.init_cov = np.zeros((emd.N, emd.N+1, emd.N+1))
    for i in range(emd.N):
        diff = emd.theta_s[0, i] - emd.init_theta[i]
        emd.init_cov[i] = emd.sigma_s[0, i] + np.outer(diff, diff)
    return emd.init_cov

def _compute_raw_Q(emd):
    """Accumulate raw Q matrix for each neuron from smoothed estimates.

    :returns: tuple (Q_raw, has_transitions)
        Q_raw: numpy.ndarray of shape (N, N+1, N+1)
        has_transitions: bool, False when T <= 1
    """
    if emd.T <= 1:
        return np.zeros((emd.N, emd.N+1, emd.N+1)), False
    # diff: (T-1, N, N+1)
    diff = emd.theta_s[1:] - emd.theta_s[:emd.T-1]
    # dd: (N, N+1, N+1) — outer products contracted over time
    dd = np.einsum('tni,tnj->nij', diff, diff)
    # Covariance terms
    lag = emd.lag_one_covariance[:emd.T-1]       # (T-1, N, N+1, N+1)
    cov_sum = (emd.sigma_s[1:emd.T] + emd.sigma_s[:emd.T-1]
               - lag - np.swapaxes(lag, -2, -1)).sum(axis=0)
    Q_raw = (dd + cov_sum) / (emd.T - 1)
    return Q_raw, True

def get_full_Q(emd):
    """
    Computes a fully dense state covariance matrix Q for each neuron.

    :param emd: container.EMData
        The data structure with smoothed parameters (theta_s, sigma_s).
    :returns: numpy.ndarray
        The updated state_cov of shape (N, N+1, N+1).
    """
    Q_raw, has_transitions = _compute_raw_Q(emd)
    if has_transitions:
        for i in range(emd.N):
            Q_raw[i] = (Q_raw[i] + Q_raw[i].T) / 2
        emd.state_cov = Q_raw
    emd.dim_param = (
        ((emd.N+1)*(emd.N+1 - 1)/2 + (emd.N+1)
         + (emd.N+1)*(emd.N+1 - 1)/2) * emd.N
    )
    return emd.state_cov

def get_scalar_Q(emd):
    """
    Computes a single scalar q (isotropic) for each neuron and updates the state covariance.

    :param emd: container.EMData
        The data structure with smoothed parameters (theta_s, sigma_s).
    :returns: numpy.ndarray
        The updated state_cov of shape (N, N+1, N+1),
        where each neuron's Q is a scalar times the identity matrix.
    """
    Q_raw, has_transitions = _compute_raw_Q(emd)
    for i in range(emd.N):
        if has_transitions:
            Q_sym = (Q_raw[i] + Q_raw[i].T) / 2
        else:
            Q_sym = Q_raw[i]
        qi = np.trace(Q_sym) / (emd.N + 1)
        emd.state_cov[i] = qi * np.eye(emd.N + 1)
    emd.dim_param = (
        (1 + (emd.N+1) + (emd.N+1)*(emd.N+1 - 1)/2) * emd.N
    )
    return emd.state_cov

def get_diagonal_Q(emd):
    """
    Computes a diagonal state covariance matrix Q for each neuron by averaging each diagonal element.

    :param emd: container.EMData
        The data structure with smoothed parameters (theta_s, sigma_s).
    :returns: numpy.ndarray
        The updated state_cov of shape (N, N+1, N+1),
        where each neuron's Q is diagonal.
    """
    Q_raw, has_transitions = _compute_raw_Q(emd)
    Q = np.zeros((emd.N, emd.N+1, emd.N+1))
    for i in range(emd.N):
        if has_transitions:
            Q[i] = np.diag(np.diag(Q_raw[i]))
        else:
            Q[i] = emd.state_cov[i]
    emd.state_cov = Q
    emd.dim_param = (
        ((emd.N+1) + (emd.N+1) + (emd.N+1)*(emd.N+1 - 1)/2) * emd.N
    )
    return emd.state_cov

def compute_eta_G(theta, F1):
    """
    Computes the first derivative (eta) and the Fisher information matrix (G).

    :param numpy.ndarray theta:
        Current parameter estimates of shape (N, N+1).
    :param numpy.ndarray F1:
        Feature matrix of shape (R, N+1), with leading ones column.

    :returns: tuple
        (eta, G)
        eta: numpy.ndarray of shape (N, N+1)
        G: numpy.ndarray of shape (N, N+1, N+1)
    """
    if _HAS_JAX:
        eta, G = _compute_eta_G_jax(jnp.asarray(theta), jnp.asarray(F1))
        return np.asarray(eta), np.asarray(G)
    # r: (R, N) — sigmoid via BLAS dgemm + SIMD vectorized exp
    r = 1 / (1 + np.exp(-F1 @ theta.T))
    # eta: (N, N+1) — BLAS dgemm
    eta = r.T @ F1
    # G: (N, N+1, N+1) — loop over N neurons, each uses BLAS dgemm
    w = r * (1 - r)
    N, Np1 = theta.shape
    G = np.empty((N, Np1, Np1))
    for n in range(N):
        wF = F1 * w[:, n:n+1]       # (R, N+1) broadcast
        G[n] = wF.T @ F1            # (N+1, N+1)
    return eta, G

def e_step_filter(emd):
    """
    Filters the parameter estimates forward in time using Newton-Raphson updates (Einstein summation).
    Complexity: O(N * R * T)

    :param emd: container.EMData
        The data structure with all necessary parameters and spike data.
    :returns: tuple
        Updated (theta_f, sigma_f, sigma_f_i, sigma_o, sigma_o_i).
    """
    if _HAS_JAX:
        results = _e_step_filter_jax(
            jnp.asarray(emd.spikes[:emd.T]),
            jnp.asarray(emd.FSUM),
            jnp.asarray(emd.init_theta),
            jnp.asarray(emd.init_cov),
            jnp.asarray(emd.state_cov),
            emd.R,
            GA_CONVERGENCE,
            MAX_GA_ITERATIONS,
        )
        theta_f, sigma_f, theta_o, sigma_o, sigma_o_i, iters = results
        if int(jnp.max(iters)) >= MAX_GA_ITERATIONS:
            raise Exception(
                "The gradient-ascent algorithm did not converge before "
                "reaching the maximum number of iterations."
            )
        emd.theta_f[:] = np.asarray(theta_f)
        emd.sigma_f[:] = np.asarray(sigma_f)
        emd.theta_o[:] = np.asarray(theta_o)
        emd.sigma_o[:] = np.asarray(sigma_o)
        emd.sigma_o_i[:] = np.asarray(sigma_o_i)
    else:
        # Pre-allocate F1 buffer to avoid repeated concatenation.
        F1 = np.empty((emd.R, emd.N + 1))
        F1[:, 0] = 1.0

        for t in range(emd.T):
            if t == 0:
                emd.theta_o[0] = emd.init_theta
                emd.sigma_o[0] = emd.init_cov
                emd.sigma_o_i[0] = np.linalg.inv(emd.sigma_o[0])
            else:
                emd.theta_o[t] = emd.theta_f[t - 1]
                emd.sigma_o[t] = emd.sigma_f[t - 1] + emd.state_cov
                emd.sigma_o_i[t] = np.linalg.inv(emd.sigma_o[t])

            # Fill F1 in-place (no allocation per time step).
            F1[:, 1:] = emd.spikes[t]

            max_dlpo = np.inf
            iterations = 0

            while max_dlpo > GA_CONVERGENCE:
                eta, G = compute_eta_G(emd.theta_f[t], F1)
                diff = emd.theta_f[t] - emd.theta_o[t]
                dlpo = eta - emd.FSUM[t] + np.matmul(
                    emd.sigma_o_i[t], diff[..., np.newaxis]
                ).squeeze(-1)
                ddlpo = G + emd.sigma_o_i[t]
                ddlpo_i = np.linalg.inv(ddlpo)

                # Update theta_f
                emd.theta_f[t] -= np.matmul(
                    ddlpo_i, dlpo[..., np.newaxis]
                ).squeeze(-1)

                max_dlpo = np.amax(np.absolute(dlpo)) / emd.R
                iterations += 1

                if iterations == MAX_GA_ITERATIONS:
                    raise Exception(
                        "The gradient-ascent algorithm did not converge before "
                        "reaching the maximum number of iterations."
                    )

                emd.sigma_f[t] = ddlpo_i

    return emd.theta_f, emd.sigma_f, emd.sigma_o, emd.sigma_o_i

@njit
def compute_eta_G_parallel(F1, theta):
    """
    Computes the first derivative (eta) and Fisher matrix (G) in a JIT-compiled manner for a single neuron.

    :param numpy.ndarray F1:
        Feature matrix of shape (R, n_params).
    :param numpy.ndarray theta:
        Parameter vector of shape (n_params,).

    :returns: tuple
        (eta, G)
        eta: numpy.ndarray of shape (n_params,)
        G: numpy.ndarray of shape (n_params, n_params)
    """
    R = F1.shape[0]
    temp = np.dot(F1, theta)
    r = np.empty(R)
    for i in range(R):
        r[i] = 1.0 / (1.0 + np.exp(-temp[i]))
    eta = np.dot(r, F1)
    p = r * (1.0 - r)
    G = np.dot(F1.T * p, F1)
    return eta, G

@njit
def process_single_i(theta_f_t_i, sigma_o_i_t_i, theta_o_t_i, FSUM_t_i,
                     F1, GA_CONVERGENCE, MAX_GA_ITERATIONS, R):
    """
    Performs Newton-Raphson updates for a single neuron's parameter vector using JIT compilation.

    :param numpy.ndarray theta_f_t_i:
        Current parameter vector of shape (n_params,).
    :param numpy.ndarray sigma_o_i_t_i:
        Inverse predicted covariance matrix of shape (n_params, n_params).
    :param numpy.ndarray theta_o_t_i:
        Predicted parameter vector (n_params,).
    :param numpy.ndarray FSUM_t_i:
        Feature sum array for the current neuron (n_params,).
    :param numpy.ndarray F1:
        Feature matrix (R, n_params).
    :param float GA_CONVERGENCE:
        Convergence threshold for gradient ascent updates.
    :param int MAX_GA_ITERATIONS:
        Maximum allowed iterations for gradient updates.
    :param int R:
        Number of trials.

    :returns: tuple
        (updated_theta, updated_inverse_hessian)
        updated_theta: numpy.ndarray of shape (n_params,)
        updated_inverse_hessian: numpy.ndarray of shape (n_params, n_params)
    """
    max_dlpo = np.inf
    iterations = 0
    while max_dlpo > GA_CONVERGENCE:
        eta, G = compute_eta_G_parallel(F1, theta_f_t_i)
        dlpo = -(FSUM_t_i - eta) + np.dot(sigma_o_i_t_i, (theta_f_t_i - theta_o_t_i))
        ddlpo = G + sigma_o_i_t_i

        L = np.linalg.cholesky(ddlpo)
        y = np.linalg.solve(L, dlpo)
        x = np.linalg.solve(L.T, y)
        theta_f_t_i -= x

        max_dlpo = np.amax(np.absolute(dlpo)) / R
        iterations += 1
        if iterations == MAX_GA_ITERATIONS:
            raise Exception(
                "Gradient-ascent algorithm did not converge within "
                "MAX_GA_ITERATIONS."
            )

    # After convergence, compute the inverse Hessian via Cholesky factor
    I = np.eye(L.shape[0])
    L_inv = np.linalg.solve(L, I)
    ddlpo_i = np.dot(L_inv.T, L_inv)  # (L^-1).T * L^-1

    return theta_f_t_i, ddlpo_i

def e_step_filter_parallel(emd):
    """
    Performs forward filtering on each neuron in parallel using joblib.

    :param emd: container.EMData
        The data structure with model parameters, spike data, etc.
    :returns: tuple
        Updated (theta_f, sigma_f, sigma_f_i, sigma_o, sigma_o_i).
    """
    GA_CONVERGENCE = 1e-5
    MAX_GA_ITERATIONS = 1000

    T = emd.T
    N = emd.N
    R = emd.R

    # Parameter dimension
    n_params = emd.theta_f[0, 0].shape[0]

    # Pre-allocate the feature matrix F1 for a single time step
    F1 = np.empty((R, n_params), dtype=emd.theta_f.dtype)
    F1[:, 0] = 1.0  # constant bias

    for t in range(T):
        if t == 0:
            emd.theta_o[0] = emd.init_theta
            emd.sigma_o[0] = emd.init_cov
            emd.sigma_o_i[0] = np.linalg.inv(emd.sigma_o[0])
        else:
            emd.theta_o[t] = emd.theta_f[t - 1]
            emd.sigma_o[t] = emd.sigma_f[t - 1] + emd.state_cov
            emd.sigma_o_i[t] = np.linalg.inv(emd.sigma_o[t])

        # Construct F1 by adding spikes data
        F1[:, 1:] = emd.spikes[t, :R]

        # Parallel processing over neurons
        results = Parallel(n_jobs=-1)(
            delayed(process_single_i)(
                emd.theta_f[t, i],
                emd.sigma_o_i[t, i],
                emd.theta_o[t, i],
                emd.FSUM[t, i],
                F1,
                GA_CONVERGENCE,
                MAX_GA_ITERATIONS,
                R
            )
            for i in range(N)
        )

        # Store results
        for i, (theta_f_i, ddlpo_i) in enumerate(results):
            emd.theta_f[t, i] = theta_f_i
            emd.sigma_f[t, i] = ddlpo_i

    return emd.theta_f, emd.sigma_f, emd.sigma_o, emd.sigma_o_i

def e_step_smooth(emd):
    """
    Performs backward smoothing on the filtered estimates (non-parallel version).
    Complexity: O(T * N)

    :param emd: container.EMData
        Data structure containing filtered parameters, covariances, etc.
    :returns: tuple
        (sigma_s, theta_o, theta_s, lag_one_covariance, A)
        representing the smoothed covariances, predicted parameters,
        smoothed parameters, lag-one covariance, and transition matrix A.
    """
    if _HAS_JAX:
        results = _e_step_smooth_jax(
            jnp.asarray(emd.theta_f),
            jnp.asarray(emd.sigma_f),
            jnp.asarray(emd.theta_o),
            jnp.asarray(emd.sigma_o),
            jnp.asarray(emd.sigma_o_i),
        )
        theta_s, sigma_s, A, lag_one_cov = results
        emd.theta_s[:] = np.asarray(theta_s)
        emd.sigma_s[:] = np.asarray(sigma_s)
        if emd.T > 1:
            emd.A[:] = np.asarray(A)
            emd.lag_one_covariance[:emd.T - 1] = np.asarray(lag_one_cov)
    else:
        emd.theta_s[emd.T - 1] = emd.theta_f[emd.T - 1]
        emd.sigma_s[emd.T - 1] = emd.sigma_f[emd.T - 1]
        for tt in range(emd.T - 1):
            t = emd.T - 2 - tt
            # A[t] = sigma_f[t] @ sigma_o_i[t+1]  — batched over N neurons
            emd.A[t] = np.matmul(emd.sigma_f[t], emd.sigma_o_i[t + 1])

            diff_theta = emd.theta_s[t + 1] - emd.theta_o[t + 1]  # (N, N+1)
            emd.theta_s[t] = emd.theta_f[t] + np.matmul(
                emd.A[t], diff_theta[..., np.newaxis]
            ).squeeze(-1)

            diff_sigma = emd.sigma_s[t + 1] - emd.sigma_o[t + 1]  # (N, N+1, N+1)
            tmp = np.matmul(np.matmul(emd.A[t], diff_sigma),
                            emd.A[t].swapaxes(-2, -1))
            emd.sigma_s[t] = emd.sigma_f[t] + tmp

            emd.lag_one_covariance[t] = np.matmul(emd.A[t], emd.sigma_s[t + 1])

    return emd.sigma_s, emd.theta_o, emd.theta_s, emd.lag_one_covariance, emd.A

def process_single_i_smoothing(sigma_f_t_i, sigma_o_i_t1_i, theta_s_t1_i,
                               theta_o_t1_i, theta_f_t_i, sigma_s_t1_i, A_t_i):
    """
    Performs smoothing updates for a single neuron in the backward pass.

    :param numpy.ndarray sigma_f_t_i:
        Filtered covariance at time t for neuron i, shape (N+1, N+1).
    :param numpy.ndarray sigma_o_i_t1_i:
        Inverse of predicted covariance at time t+1 for neuron i, shape (N+1, N+1).
    :param numpy.ndarray theta_s_t1_i:
        Smoothed parameters at time t+1 for neuron i, shape (N+1,).
    :param numpy.ndarray theta_o_t1_i:
        Predicted parameters at time t+1 for neuron i, shape (N+1,).
    :param numpy.ndarray theta_f_t_i:
        Filtered parameters at time t for neuron i, shape (N+1,).
    :param numpy.ndarray sigma_s_t1_i:
        Smoothed covariance at time t+1 for neuron i, shape (N+1, N+1).
    :param numpy.ndarray A_t_i:
        Transition matrix from time t to t+1 for neuron i, shape (N+1, N+1).

    :returns: tuple
        (theta_s_t_i, sigma_s_t_i, A_t_i, lag_one_covariance_t_i)
        representing the updated smoothed parameters, smoothed covariance,
        transition matrix, and lag-one covariance for neuron i at time t.
    """
    A_t_i = np.dot(sigma_f_t_i, sigma_o_i_t1_i)
    tmp_theta = np.dot(A_t_i, (theta_s_t1_i - theta_o_t1_i))
    theta_s_t_i = theta_f_t_i + tmp_theta

    tmp_sigma = np.dot(A_t_i, (sigma_s_t1_i - sigma_o_i_t1_i))
    tmp_sigma = np.dot(tmp_sigma, A_t_i.T)
    sigma_s_t_i = sigma_f_t_i + tmp_sigma

    lag_one_covariance_t_i = np.dot(A_t_i, sigma_s_t1_i)
    return theta_s_t_i, sigma_s_t_i, A_t_i, lag_one_covariance_t_i

def e_step_smooth_parallel(emd):
    """
    Performs backward smoothing on filtered estimates using parallel processing for each neuron.

    :param emd: container.EMData
        Data structure containing filtered parameters, covariances, etc.
    :returns: tuple
        (sigma_s, theta_o, theta_s, lag_one_covariance, A)
    """
    emd.theta_s[emd.T - 1] = emd.theta_f[emd.T - 1]
    emd.sigma_s[emd.T - 1] = emd.sigma_f[emd.T - 1]

    for tt in range(emd.T - 1):
        t = emd.T - 2 - tt

        results = Parallel(n_jobs=-1)(
            delayed(process_single_i_smoothing)(
                emd.sigma_f[t, i],
                emd.sigma_o_i[t + 1, i],
                emd.theta_s[t + 1, i],
                emd.theta_o[t + 1, i],
                emd.theta_f[t, i],
                emd.sigma_s[t + 1, i],
                emd.A[t, i]
            )
            for i in range(emd.N)
        )

        for i, (theta_s_t_i, sigma_s_t_i, A_t_i, lag_one_covariance_t_i) in enumerate(results):
            emd.theta_s[t, i] = theta_s_t_i
            emd.sigma_s[t, i] = sigma_s_t_i
            emd.A[t, i] = A_t_i
            emd.lag_one_covariance[t, i] = lag_one_covariance_t_i

    return emd.sigma_s, emd.theta_o, emd.theta_s, emd.lag_one_covariance, emd.A
