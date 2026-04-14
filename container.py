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
from .probability import log_marginal

class EMData:
    def __init__(self, spikes, state_cov=0.5, u=None, v=None):
        """
        Container class for storing and managing the results of the EM algorithm
        for the state-space kinetic Ising model. This class holds the spike data,
        parameters, covariances, and timing information required for filtering,
        smoothing, and parameter estimation over time.

        The complexity of the initialization is O(1) with respect to the model fitting,
        although loops are used to set up arrays for each neuron.

        :param numpy.ndarray spikes:
            A binary spike array with dimensions (T+1, R, N) where:
              - T+1: number of time bins (including an initial time t=0),
              - R: number of trials (or runs),
              - N: number of neurons.
              A value of 1 indicates a spike and 0 indicates no spike.
        :param state_cov:
            Controls the Q (state covariance) estimation method:
              - scalar (int/float): Q[i] = state_cov * I(N+1), updated via get_scalar_Q
              - vector shape (N+1,): Q[i] = diag(state_cov), updated via get_diagonal_Q
              - matrix shape (N+1, N+1): Q[i] = state_cov, updated via get_full_Q
              - 0 or None: Q[i] = zeros, no Q update (fixed)
            Default is 0.5.

        Attributes:
            spikes (numpy.ndarray): Input spike data.
            T (int): Number of time bins used in the model (spikes.shape[0] - 1).
            R (int): Number of trials (spikes.shape[1]).
            N (int): Number of neurons (spikes.shape[2]).
            dim_param (int): Dimension of the parameter vector (initialized to 0).
            aic (float): Akaike Information Criterion (initialized to 0).

            state_cov (numpy.ndarray): An array of shape (N, N+1, N+1) containing the
                state covariance matrices for each neuron. Initialized to 0.5 * identity.
            F (numpy.ndarray): An array of shape (N, N+1, N+1) representing the state
                transition matrices for each neuron. Initialized to identity.
            init_cov (numpy.ndarray): Initial covariance matrices for each neuron
                (shape: (N, N+1, N+1)), initialized to identity.
            init_cov_i (numpy.ndarray): Inverse of the initial covariance matrices.
            init_theta (numpy.ndarray): Initial parameter (theta) estimates for each neuron
                (shape: (N, N+1)), initialized to zeros.

            e_step_time, filtering_time, smoothing_time, m_step_time, llk_time (float):
                Timing metrics for the E-step, filtering, smoothing, M-step, and log likelihood
                computation.
            FSUM (numpy.ndarray): Pre-computed array of shape (T, N, N+1) used for accumulating
                spike-based sums. Computed over trials and time.

            theta_o (numpy.ndarray): "Observation" theta parameters of shape (T, N, N+1).
            theta_f (numpy.ndarray): Filtered theta estimates (shape: (T, N, N+1)).
            theta_s (numpy.ndarray): Smoothed theta estimates (shape: (T, N, N+1)).
            eta_s (numpy.ndarray): Smoothed expectation parameters (shape: (T, N, N+1)).

            sigma_o (numpy.ndarray): Observation covariance matrices
                (shape: (T, N, N+1, N+1)).
            sigma_f (numpy.ndarray): Filtered covariance matrices
                (shape: (T, N, N+1, N+1)).
            sigma_f_i (numpy.ndarray): Inverse of filtered covariance matrices.
            sigma_o_i (numpy.ndarray): Inverse of observation covariance matrices.
            sigma_s (numpy.ndarray): Smoothed covariance matrices (shape: (T, N, N+1, N+1)),
                initialized to ones.

            A (numpy.ndarray): Lag-one state transition matrices (shape: (T-1, N, N+1, N+1)).
            lag_one_covariance (numpy.ndarray): Lag-one covariance matrices
                (shape: (T, N, N+1, N+1)).

            marg_llk (function): Function pointer to compute the marginal log likelihood.
            mll (float): Current marginal log likelihood (initialized to infinity).
            mll_list (list): List to store the marginal log likelihood values over iterations.
            iterations_list (list): List to record iteration numbers.
            Q_list (list): List to store Q matrices (state covariance) over iterations.
            F_list (list): List to store F matrices over iterations.
            iterations (int): Counter for the number of EM iterations.
            CONVERGED (float): Convergence threshold (default: 1e-5).
            convergence (float): Current convergence value (initialized to infinity).
        """
        self.spikes = spikes
        self.T = spikes.shape[0] - 1
        self.R = spikes.shape[1]
        self.N = spikes.shape[2]

        # Exogenous input
        if u is not None:
            self.u = u
            self.d_u = u.shape[1]
            self.U = np.zeros((self.N, self.N + 1, self.d_u))
        else:
            self.u = None
            self.d_u = 0
            self.U = None

        # Observation input — accept (T, d_v) or (T, R, d_v)
        if v is not None:
            v = np.asarray(v)
            if v.ndim == 2:
                # Broadcast (T, d_v) → (T, R, d_v)
                v = np.broadcast_to(v[:, np.newaxis, :],
                                    (v.shape[0], self.R, v.shape[1])).copy()
            self.v = v              # (T, R, d_v)
            self.d_v = v.shape[-1]
            self.V = np.zeros((self.N, self.d_v))
        else:
            self.v = None
            self.d_v = 0
            self.V = None

        self.dim_param = 0
        self.aic = 0

        # Store original state_cov input for m_step dispatch.
        self.state_cov_0 = state_cov

        # Initialize state covariance matrices based on state_cov type.
        eye = np.eye(self.N + 1)
        self.state_cov = np.zeros((self.N, self.N+1, self.N+1))
        if state_cov is None or (np.isscalar(state_cov) and state_cov == 0):
            pass  # Q[i] = zeros (no dynamics)
        elif np.isscalar(state_cov):
            self.state_cov[:] = state_cov * eye
        else:
            state_cov = np.asarray(state_cov)
            if state_cov.ndim == 1:
                self.state_cov[:] = np.diag(state_cov)
            elif state_cov.ndim == 2:
                self.state_cov[:] = state_cov

        # Initialize F matrices as identity for each neuron.
        self.F = np.tile(eye, (self.N, 1, 1))

        # Initialize initial covariance matrices as identity.
        self.init_cov = np.tile(eye, (self.N, 1, 1))
        self.init_cov_i = np.tile(eye, (self.N, 1, 1))

        # Initialize initial theta parameters as zeros.
        self.init_theta = np.zeros((self.N, self.N+1))

        # Timing metrics.
        self.e_step_time = 0
        self.filtering_time = 0
        self.smoothing_time = 0
        self.m_step_time = 0
        self.llk_time = 0

        # Pre-compute FSUM: an array that sums spike activity and their products across trials.
        current = self.spikes[1:]          # (T, R, N)
        prev = self.spikes[:self.T]        # (T, R, N)
        self.FSUM = np.zeros((self.T, self.N, self.N + 1))
        self.FSUM[:, :, 0] = current.sum(axis=1)
        self.FSUM[:, :, 1:] = np.einsum('trn,trm->tnm', current, prev)
        # Initialize theta estimates.
        self.theta_o = np.zeros((self.T, self.N, self.N+1))
        self.theta_f = np.zeros((self.T, self.N, self.N+1))
        self.theta_s = np.zeros((self.T, self.N, self.N+1))
        self.eta_s = np.zeros((self.T, self.N, self.N+1))  # Smoothed expectations.

        # Initialize covariance estimates.
        self.sigma_f = np.tile(eye, (self.T, self.N, 1, 1))
        self.sigma_o = np.tile(eye, (self.T, self.N, 1, 1))
        self.sigma_o_i = np.tile(eye, (self.T, self.N, 1, 1))
        self.sigma_s = np.ones((self.T, self.N, self.N+1, self.N+1))

        # Initialize transition and lag-one covariance matrices.
        self.A = np.zeros((self.T-1, self.N, self.N+1, self.N+1))
        self.lag_one_covariance = np.zeros((self.T, self.N, self.N+1, self.N+1))

        # Set the marginal log likelihood function.
        self.marg_llk = log_marginal
        self.mll = np.inf
        self.mll_list = []
        self.iterations_list = []
        self.Q_list = []
        self.F_list = []
        self.iterations = 0
        self.CONVERGED = 1e-5
        self.convergence = np.inf
