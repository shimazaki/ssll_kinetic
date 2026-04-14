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

__version__ = "0.1.0"

import numpy as np
import time
from tqdm import tqdm

from . import container
from . import exp_max
from .probability import log_marginal

def run(spikes, max_iter=100, mstep=True, state_cov=0.5, stationary=False,
        EM_Info=True, u=None, v=None):
    """
    Runs the Expectation-Maximization (EM) algorithm to fit the state-space kinetic Ising model
    to spike data.

    :param numpy.ndarray spikes:
        A binary array of shape (T+1, R, N) representing spike data, where T is the number of
        time bins, R is the number of trials, and N is the number of neurons. A value of 1
        indicates a spike, and 0 indicates no spike.
    :param int max_iter:
        The maximum number of EM iterations to perform. Default is 100.
    :param bool mstep:
        Whether to perform the maximization step (M-step) at each iteration. Default is True.
    :param state_cov:
        Controls the Q (state covariance) estimation method in the M-step:
          - scalar (int/float): isotropic Q, updated via get_scalar_Q. Default is 0.5.
          - vector shape (N+1,): diagonal Q, updated via get_diagonal_Q
          - matrix shape (N+1, N+1): full dense Q, updated via get_full_Q
          - 0 or None: fixed Q (no update)
        Ignored when stationary=True (forced to 0).
    :param bool stationary:
        If True, fit a time-independent model by pooling all T*R transition
        observations into a single time step. Spikes (T+1, R, N) are reshaped
        to (2, T*R, N) and state_cov is forced to 0. Default is False.

        .. note:: When ``stationary=True`` with exogenous input ``u`` or ``v``,
           the input is time-averaged into a single vector, losing temporal
           variation. To fit a stationary model that preserves time-varying
           input (e.g., stimulus history), use ``state_cov=0`` instead:
           this keeps all T time steps with their per-step input while
           constraining theta to be constant (zero state noise).
           Example: ``emd = ssll_kinetic.run(spikes, state_cov=0, v=v)``
    :param bool EM_Info:
        If True, display a tqdm progress bar during EM iterations and print
        final results. Default is True.
    :param numpy.ndarray u:
        State input array of shape (T, d_u). When provided, the state equation
        becomes theta_t = theta_{t-1} + U·u_t + xi_t and U is learned via the
        M-step. Default is None.
    :param numpy.ndarray v:
        Observation input array of shape (T, d_v). When provided, the firing
        rate includes an additive offset V·v_t and V is learned via
        Newton-Raphson in the M-step. Default is None.

        .. note:: With ``stationary=True``, ``v`` is time-averaged into a
           constant offset indistinguishable from the bias. To retain
           per-time-step observation input while fitting stationary parameters,
           use ``state_cov=0`` instead of ``stationary=True``.

    :returns:
        container.EMData
            An EMData object containing:
              - The posterior parameter estimates (emd.theta_s).
              - The log marginal likelihood for each iteration (emd.mll_list).
              - The sequence of state covariance matrices (emd.Q_list).
              - The total number of completed iterations (emd.iterations).
              - Timing information for the E-step, M-step, and log-likelihood calculations.
              - The Akaike Information Criterion (AIC) based on the final likelihood.
    """
    if stationary:
        T_plus_1, R, N = spikes.shape
        T = T_plus_1 - 1
        from_states = spikes[:T].reshape(T * R, N)
        to_states = spikes[1:].reshape(T * R, N)
        spikes = np.stack([from_states, to_states])  # (2, T*R, N)
        state_cov = 0
        if u is not None:
            import warnings
            warnings.warn(
                "stationary=True with u: time-averaging u over T steps. "
                "U is not identifiable with T=1; it will remain at zero.",
                stacklevel=2)
            u = u.mean(axis=0, keepdims=True)
        if v is not None:
            import warnings
            warnings.warn(
                "stationary=True with v: time-averaging v over T steps. "
                "V acts as a constant offset indistinguishable from bias.",
                stacklevel=2)
            v = v.mean(axis=0, keepdims=True)

    # Initialize the EMData container with the given spike data
    emd = container.EMData(spikes, state_cov=state_cov, u=u, v=v)

    # Compute the initial marginal log likelihood
    lmc = emd.marg_llk(emd)
    mll = np.inf

    # Initialize iteration count
    emd.iterations = 0

    # EM loop
    pbar = tqdm(total=max_iter, desc='EM', disable=not EM_Info)
    while emd.iterations < max_iter and (emd.convergence > emd.CONVERGED):
        pbar.set_postfix(conv=f'{emd.convergence:.6f}')

        # E-step timing
        t0 = time.perf_counter()
        exp_max.e_step(emd)
        emd.e_step_time = time.perf_counter() - t0

        # M-step timing (only if mstep=True)
        if mstep:
            t0 = time.perf_counter()
            exp_max.m_step(emd)
            emd.m_step_time = time.perf_counter() - t0

        lmp = lmc
        t0 = time.perf_counter()
        lmc = emd.marg_llk(emd)
        emd.llk_time = time.perf_counter() - t0

        emd.mll_list.append(lmc)
        emd.mll = lmc
        emd.Q_list.append(emd.state_cov)
        emd.iterations_list.append(emd.iterations)

        emd.iterations += 1
        emd.convergence = (lmp - lmc) / lmp if lmp != 0 else 0.0
        pbar.update(1)
    pbar.close()

    # Compute AIC after finishing
    if stationary:
        emd.dim_param = emd.N * (emd.N + 1)
    if emd.U is not None and emd.T > 1:
        emd.dim_param += emd.N * (emd.N + 1) * emd.d_u
    if emd.V is not None:
        emd.dim_param += emd.N * emd.d_v
    emd.aic = -2 * emd.mll + 2 * emd.dim_param
    if EM_Info:
        print('Log marginal likelihood = %.6f (%d iterations)' %
              (emd.mll, emd.iterations))

    return emd
