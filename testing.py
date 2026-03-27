"""
Functions for testing results.

---

This code tests the state-space kinetic Ising model described in:
Ken Ishihara, Hideaki Shimazaki. *State-space kinetic Ising model reveals
task-dependent entropy flow in sparsely active nonequilibrium neuronal
dynamics*. (2025) arXiv:2502.15440

The implementation extends existing libraries available at:
- https://github.com/christiando/ssll_lib.git
- https://github.com/shimazaki/dynamic_corr

Copyright (C) 2025
Authors of the extensions: Ken Ishihara (KenIshihara-17171ken)
                           Hideaki Shimazaki (h.shimazaki@i.kyoto-u.ac.jp)

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

import unittest
import time

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import ssll_kinetic
from ssll_kinetic import synthesis, container, exp_max, entropy_flow, probability

# Test Parameters
DEFAULT_T = 20   # Number of time steps
DEFAULT_R = 20   # Number of trials
DEFAULT_N = 3    # Number of neurons
DEFAULT_THETA_SEED = 42   # Random seed for theta generation
DEFAULT_SPIKE_SEED = 1    # Random seed for spike generation
DEFAULT_MLLK_TOLERANCE = 1e-6  # Tolerance for log marginal likelihood comparison
DEFAULT_ENTROPY_TOLERANCE = 1e-4  # Tolerance for entropy flow comparison

# Expected Spike Count
EXPECTED_SPIKE_COUNT = 1094  # Total spikes for T=20, R=20, N=3

# Expected Log Marginal Likelihood Values
EXPECTED_MLLK_DIAGONAL_Q = -261.296332
EXPECTED_MLLK_FULL_Q = -256.209241
EXPECTED_MLLK_SCALAR_Q = -263.798103
EXPECTED_MLLK_NO_MSTEP = -278.074169

# Edge Case Expected Values
EXPECTED_MLLK_SINGLE_NEURON = -142.615633
EXPECTED_MLLK_SINGLE_TRIAL = -17.170574

# Expected Entropy Flow Values (diagonal Q, T=20, R=20, N=3)
EXPECTED_TOTAL_FORWARD_ENTROPY = 11.493761
EXPECTED_TOTAL_REVERSE_ENTROPY = 14.172967
EXPECTED_TOTAL_NET_ENTROPY_FLOW = 2.679206


def generate_test_data(T, R, N, theta_seed=DEFAULT_THETA_SEED,
                       spike_seed=DEFAULT_SPIKE_SEED):
    """
    Generates reproducible test data (theta and spikes) with fixed seeds.

    Args:
        T: Number of time bins
        R: Number of trials
        N: Number of neurons
        theta_seed: Random seed for theta generation
        spike_seed: Random seed for spike generation
    Returns:
        (THETA, spikes) tuple
    """
    np.random.seed(theta_seed)
    THETA = synthesis.generate_thetas(T, N)
    np.random.seed(spike_seed)
    spikes = synthesis.generate_spikes(T, R, N, THETA)
    return THETA, spikes


def run_em_with_q_method(spikes, q_method='diagonal', max_iter=100):
    """
    Runs the EM algorithm with a specified Q estimation method via state_cov.

    Args:
        spikes: Spike data array of shape (T+1, R, N)
        q_method: One of 'diagonal', 'full', 'scalar'
        max_iter: Maximum number of EM iterations
    Returns:
        EMData object containing results
    """
    N = spikes.shape[2]
    state_cov_map = {
        'scalar': 0.5,
        'diagonal': 0.5 * np.ones(N + 1),
        'full': 0.5 * np.identity(N + 1),
    }
    state_cov = state_cov_map[q_method]
    return ssll_kinetic.run(spikes, max_iter=max_iter, state_cov=state_cov)


class TestEstimator(unittest.TestCase):
    """
    Tests for the state-space kinetic Ising model estimator.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.T = DEFAULT_T
        self.R = DEFAULT_R
        self.N = DEFAULT_N

    def test_0_spike_generation(self):
        """Test that spike generation is deterministic and produces expected counts."""
        print("Test Spike Generation.")
        start_cpu_time = time.process_time()

        THETA, spikes = generate_test_data(self.T, self.R, self.N)
        spike_count = int(np.sum(spikes))

        print('Spike Count = %d (expected %d)' % (spike_count, EXPECTED_SPIKE_COUNT))
        self.assertEqual(spike_count, EXPECTED_SPIKE_COUNT)

        # Check shape
        self.assertEqual(spikes.shape, (self.T + 1, self.R, self.N))
        # Check binary values
        self.assertTrue(np.all((spikes == 0) | (spikes == 1)))

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_1_diagonal_Q(self):
        """Test EM with diagonal state covariance estimation (default)."""
        print("Test Diagonal Q Estimation (T=%d, R=%d, N=%d)." %
              (self.T, self.R, self.N))
        start_cpu_time = time.process_time()

        THETA, spikes = generate_test_data(self.T, self.R, self.N)
        emd = run_em_with_q_method(spikes, q_method='diagonal')

        print('Log marginal likelihood = %.6f (expected %.6f)' %
              (emd.mllk, EXPECTED_MLLK_DIAGONAL_Q))
        self.assertFalse(
            np.absolute(emd.mllk - EXPECTED_MLLK_DIAGONAL_Q) > DEFAULT_MLLK_TOLERANCE,
            "Diagonal Q mllk mismatch: got %.6f, expected %.6f" %
            (emd.mllk, EXPECTED_MLLK_DIAGONAL_Q))

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_2_full_Q(self):
        """Test EM with full dense state covariance estimation."""
        print("Test Full Q Estimation (T=%d, R=%d, N=%d)." %
              (self.T, self.R, self.N))
        start_cpu_time = time.process_time()

        THETA, spikes = generate_test_data(self.T, self.R, self.N)
        emd = run_em_with_q_method(spikes, q_method='full')

        print('Log marginal likelihood = %.6f (expected %.6f)' %
              (emd.mllk, EXPECTED_MLLK_FULL_Q))
        self.assertFalse(
            np.absolute(emd.mllk - EXPECTED_MLLK_FULL_Q) > DEFAULT_MLLK_TOLERANCE,
            "Full Q mllk mismatch: got %.6f, expected %.6f" %
            (emd.mllk, EXPECTED_MLLK_FULL_Q))

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_3_scalar_Q(self):
        """Test EM with scalar (isotropic) state covariance estimation."""
        print("Test Scalar Q Estimation (T=%d, R=%d, N=%d)." %
              (self.T, self.R, self.N))
        start_cpu_time = time.process_time()

        THETA, spikes = generate_test_data(self.T, self.R, self.N)
        emd = run_em_with_q_method(spikes, q_method='scalar')

        print('Log marginal likelihood = %.6f (expected %.6f)' %
              (emd.mllk, EXPECTED_MLLK_SCALAR_Q))
        self.assertFalse(
            np.absolute(emd.mllk - EXPECTED_MLLK_SCALAR_Q) > DEFAULT_MLLK_TOLERANCE,
            "Scalar Q mllk mismatch: got %.6f, expected %.6f" %
            (emd.mllk, EXPECTED_MLLK_SCALAR_Q))

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_4_no_mstep(self):
        """Test EM without M-step (E-step only, fixed state covariance)."""
        print("Test No M-step (T=%d, R=%d, N=%d)." %
              (self.T, self.R, self.N))
        start_cpu_time = time.process_time()

        THETA, spikes = generate_test_data(self.T, self.R, self.N)
        emd = ssll_kinetic.run(spikes, max_iter=100, mstep=False)

        print('Log marginal likelihood = %.6f (expected %.6f)' %
              (emd.mllk, EXPECTED_MLLK_NO_MSTEP))
        self.assertFalse(
            np.absolute(emd.mllk - EXPECTED_MLLK_NO_MSTEP) > DEFAULT_MLLK_TOLERANCE,
            "No M-step mllk mismatch: got %.6f, expected %.6f" %
            (emd.mllk, EXPECTED_MLLK_NO_MSTEP))

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_5_edge_case_single_neuron(self):
        """Test with a single neuron (N=1)."""
        print("Test Single Neuron (N=1).")
        start_cpu_time = time.process_time()

        THETA, spikes = generate_test_data(self.T, self.R, N=1)
        emd = run_em_with_q_method(spikes, q_method='diagonal')

        print('Log marginal likelihood = %.6f (expected %.6f)' %
              (emd.mllk, EXPECTED_MLLK_SINGLE_NEURON))
        self.assertFalse(
            np.absolute(emd.mllk - EXPECTED_MLLK_SINGLE_NEURON) > DEFAULT_MLLK_TOLERANCE,
            "Single neuron mllk mismatch: got %.6f, expected %.6f" %
            (emd.mllk, EXPECTED_MLLK_SINGLE_NEURON))

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_6_edge_case_single_trial(self):
        """Test with a single trial (R=1)."""
        print("Test Single Trial (R=1).")
        start_cpu_time = time.process_time()

        THETA, spikes = generate_test_data(self.T, R=1, N=self.N)
        emd = run_em_with_q_method(spikes, q_method='diagonal')

        print('Log marginal likelihood = %.6f (expected %.6f)' %
              (emd.mllk, EXPECTED_MLLK_SINGLE_TRIAL))
        self.assertFalse(
            np.absolute(emd.mllk - EXPECTED_MLLK_SINGLE_TRIAL) > DEFAULT_MLLK_TOLERANCE,
            "Single trial mllk mismatch: got %.6f, expected %.6f" %
            (emd.mllk, EXPECTED_MLLK_SINGLE_TRIAL))

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_7_entropy_flow(self):
        """Test entropy flow computation from EM results."""
        print("Test Entropy Flow (T=%d, R=%d, N=%d)." %
              (self.T, self.R, self.N))
        start_cpu_time = time.process_time()

        THETA, spikes = generate_test_data(self.T, self.R, self.N)
        emd = run_em_with_q_method(spikes, q_method='diagonal')

        # Compute entropy flow
        sf_bath, sr_bath, s_bath, M = entropy_flow.compute_entropy_flow(emd)

        # Check shapes
        self.assertEqual(sf_bath.shape, (self.T, self.N))
        self.assertEqual(sr_bath.shape, (self.T, self.N))
        self.assertEqual(s_bath.shape, (self.T, self.N))
        self.assertEqual(M.shape, (self.T, self.N))

        # Check net flow identity: s_bath == -(sf_bath - sr_bath)
        self.assertTrue(np.allclose(s_bath, -(sf_bath - sr_bath)),
                        "Net entropy flow should equal -(forward - reverse)")

        # Check forward entropy is non-negative (conditional entropy)
        self.assertTrue(np.all(sf_bath >= 0),
                        "Forward conditional entropy should be non-negative")

        # Check spike probabilities are in [0, 1]
        active_M = M  # All T time steps have computed values
        self.assertTrue(np.all((active_M >= 0) & (active_M <= 1)),
                        "Mean-field spike probabilities should be in [0, 1]")

        # Check total values against expected
        total_forward = np.sum(sf_bath)
        total_reverse = np.sum(sr_bath)
        total_net = np.sum(s_bath)

        print('Total forward entropy = %.6f (expected %.6f)' %
              (total_forward, EXPECTED_TOTAL_FORWARD_ENTROPY))
        print('Total reverse entropy = %.6f (expected %.6f)' %
              (total_reverse, EXPECTED_TOTAL_REVERSE_ENTROPY))
        print('Total net entropy flow = %.6f (expected %.6f)' %
              (total_net, EXPECTED_TOTAL_NET_ENTROPY_FLOW))

        self.assertFalse(
            np.absolute(total_forward - EXPECTED_TOTAL_FORWARD_ENTROPY) > DEFAULT_ENTROPY_TOLERANCE,
            "Forward entropy mismatch: got %.6f, expected %.6f" %
            (total_forward, EXPECTED_TOTAL_FORWARD_ENTROPY))
        self.assertFalse(
            np.absolute(total_reverse - EXPECTED_TOTAL_REVERSE_ENTROPY) > DEFAULT_ENTROPY_TOLERANCE,
            "Reverse entropy mismatch: got %.6f, expected %.6f" %
            (total_reverse, EXPECTED_TOTAL_REVERSE_ENTROPY))
        self.assertFalse(
            np.absolute(total_net - EXPECTED_TOTAL_NET_ENTROPY_FLOW) > DEFAULT_ENTROPY_TOLERANCE,
            "Net entropy flow mismatch: got %.6f, expected %.6f" %
            (total_net, EXPECTED_TOTAL_NET_ENTROPY_FLOW))

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_8_convergence(self):
        """Test that EM actually converges (doesn't just hit max_iter)."""
        print("Test EM Convergence.")
        start_cpu_time = time.process_time()

        THETA, spikes = generate_test_data(self.T, self.R, self.N)
        emd = run_em_with_q_method(spikes, q_method='diagonal', max_iter=500)

        # Should converge well before 500 iterations
        print('Converged in %d iterations' % emd.iterations)
        self.assertLess(emd.iterations, 500,
                        "EM should converge before reaching max_iter=500")

        # Log marginal likelihood should be monotonically non-decreasing
        mllk_array = np.array(emd.mllk_list)
        diffs = np.diff(mllk_array)
        # Allow small numerical violations (1e-10)
        self.assertTrue(np.all(diffs >= -1e-10),
                        "Log marginal likelihood should be monotonically non-decreasing")

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_9_parameter_recovery(self):
        """Test that estimated parameters recover the true theta (high R)."""
        print("Test Parameter Recovery (high trial count).")
        start_cpu_time = time.process_time()

        T, N = 50, 2
        R = 500  # Many trials for reliable recovery

        np.random.seed(DEFAULT_THETA_SEED)
        THETA = synthesis.generate_thetas(T, N, mu=0.5, sigma=10.0)
        np.random.seed(DEFAULT_SPIKE_SEED)
        spikes = synthesis.generate_spikes(T, R, N, THETA)

        emd = run_em_with_q_method(spikes, q_method='diagonal')

        # Compare estimated theta_s with true THETA
        # With R=500, all parameters (field and coupling) should be well recovered
        for i in range(N):
            for j in range(N + 1):
                true_param = THETA[:, i, j]
                est_param = emd.theta_s[:, i, j]
                corr = np.corrcoef(true_param, est_param)[0, 1]
                label = 'field' if j == 0 else 'coupling J_%d' % j
                print('Neuron %d, %s: correlation = %.4f' % (i, label, corr))
                self.assertGreater(corr, 0.8,
                    "Parameter recovery correlation should be > 0.8 "
                    "(neuron %d, %s, got %.4f)" % (i, label, corr))

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_10_entropy_formulation_equivalence(self):
        """Test chi-based vs h-psi decomposition of conditional entropy.

        The chi-based formulation (update_S) integrates chi(h) directly (Eq. 48).
        The h-psi formulation (update_S_alt) uses the identity
        chi(h) = -[r(h)*h - psi(h)] (Eq. 47) to decompose conditional entropy
        into expected natural parameter and log-partition terms.

        Both formulations should agree to machine precision since
        E[chi(h)] = E[-r(h)*h + psi(h)] = -(E[r(h)*h] - E[psi(h)]).
        """
        print("Test Entropy Formulation Equivalence (chi vs h-psi).")
        start_cpu_time = time.process_time()

        THETA, spikes = generate_test_data(self.T, self.R, self.N)
        emd = run_em_with_q_method(spikes, q_method='diagonal')

        # Compute mean spikes and entropy using both formulations
        mp = np.mean(emd.spikes, axis=(0, 1))
        for t in range(min(emd.T - 1, 5)):  # Test first 5 time steps
            m_p = mp if t == 0 else m
            THETA_st = emd.theta_s[t]
            H = THETA_st[:, 0]
            J = np.delete(THETA_st, 0, 1)
            m = entropy_flow.update_m_P_t1_o1(H, J, m_p)

            # Chi-based (main formulation)
            S_fwd_chi = entropy_flow.update_S(H, J, m_p)
            S_rev_chi = entropy_flow.update_S_re(H, J, m, m_p)

            # h-psi decomposition (alternative formulation)
            S_fwd_alt = entropy_flow.update_S_alt(H, J, m, m_p)
            S_rev_alt = entropy_flow.update_S_re_alt(H, J, m, m_p)

            fwd_diff = np.max(np.abs(S_fwd_chi - S_fwd_alt))
            rev_diff = np.max(np.abs(S_rev_chi - S_rev_alt))

            print('t=%d: forward diff=%.2e, reverse diff=%.2e' %
                  (t, fwd_diff, rev_diff))

            self.assertLess(fwd_diff, 1e-6,
                "Forward entropy formulations should agree at t=%d "
                "(diff=%.2e)" % (t, fwd_diff))
            self.assertLess(rev_diff, 1e-6,
                "Reverse entropy formulations should agree at t=%d "
                "(diff=%.2e)" % (t, rev_diff))

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))


    def test_11_single_time_step(self):
        """Test EM with T=1 (no state transitions, Q update skipped)."""
        print("Test Single Time Step (T=1, R=20, N=3).")
        start_cpu_time = time.process_time()

        T, R, N = 1, 20, 3
        THETA, spikes = generate_test_data(T, R, N)

        # Should not crash despite T-1=0
        emd = run_em_with_q_method(spikes, q_method='diagonal')

        # theta_s shape: (T, N, N+1) — note T not T+1 for theta_s
        self.assertEqual(emd.theta_s.shape[1], N)
        self.assertEqual(emd.theta_s.shape[2], N + 1)

        # Entropy flow should return shape (T, N)
        sf_bath, sr_bath, s_bath, M = entropy_flow.compute_entropy_flow(emd)
        self.assertEqual(sf_bath.shape, (T, N))

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))


    def test_12_stationary(self):
        """Test run_stationary pools transitions and returns a single theta."""
        print("Test Stationary Analysis (T=50, R=20, N=2).")
        start_cpu_time = time.process_time()

        T, R, N = 50, 20, 2
        np.random.seed(DEFAULT_THETA_SEED)
        THETA = synthesis.generate_thetas(T, N, mu=0.5, sigma=10.0)
        np.random.seed(DEFAULT_SPIKE_SEED)
        spikes = synthesis.generate_spikes(T, R, N, THETA)

        emd = ssll_kinetic.run(spikes, max_iter=50, stationary=True)

        # Should have pooled T*R trials into a single time step
        self.assertEqual(emd.T, 1)
        self.assertEqual(emd.R, T * R)
        self.assertEqual(emd.theta_s.shape, (1, N, N + 1))

        # Q should remain zero throughout
        self.assertTrue(np.all(emd.state_cov == 0),
            "State covariance Q should remain zero for stationary model")

        # dim_pram and AIC should be set
        self.assertEqual(emd.dim_param, N * (N + 1))
        expected_aic = -2 * emd.mllk + 2 * emd.dim_param
        self.assertAlmostEqual(emd.aic, expected_aic, places=6)

        # Should converge
        self.assertLessEqual(emd.convergence, emd.CONVERGED)

        end_cpu_time = time.process_time()
        print('Stationary theta:', emd.theta_s[0])
        print('AIC: %.4f' % emd.aic)
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))


    def test_13_stationary_entropy_flow(self):
        """Test stationary entropy flow: fixed-point convergence and positivity."""
        print("Test Stationary Entropy Flow.")
        start_cpu_time = time.process_time()

        T, R, N = 50, 20, 2
        np.random.seed(DEFAULT_THETA_SEED)
        THETA = synthesis.generate_thetas(T, N, mu=0.5, sigma=10.0)
        np.random.seed(DEFAULT_SPIKE_SEED)
        spikes = synthesis.generate_spikes(T, R, N, THETA)

        emd = ssll_kinetic.run(spikes, max_iter=50, stationary=True)
        sf, sr, s_net, M = entropy_flow.compute_entropy_flow(emd)

        # Shapes should be (1, N) from unified interface
        self.assertEqual(sf.shape, (1, N))
        self.assertEqual(sr.shape, (1, N))
        self.assertEqual(s_net.shape, (1, N))
        self.assertEqual(M.shape, (1, N))

        m_star = M[0]

        # m_star should be a fixed point: m* = f(theta, m*)
        theta = emd.theta_s[0]
        m_check = entropy_flow.compute_mean_field(theta, m_star)
        self.assertLess(np.max(np.abs(m_star - m_check)), 1e-7,
            "m_star should be a fixed point of the mean-field equation")

        # Probabilities in (0, 1)
        self.assertTrue(np.all(m_star > 0) and np.all(m_star < 1))

        # Forward and reverse entropies should be positive
        self.assertTrue(np.all(sf > 0), "Forward entropy should be positive")
        self.assertTrue(np.all(sr > 0), "Reverse entropy should be positive")

        # Net entropy flow (dissipation) should be non-negative at stationarity
        self.assertTrue(np.all(s_net >= -1e-10),
            "Net entropy flow should be non-negative at stationarity")

        end_cpu_time = time.process_time()
        print('Fixed point m*:', m_star)
        print('Forward entropy:', sf[0])
        print('Reverse entropy:', sr[0])
        print('Net entropy flow:', s_net[0])
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))


    def test_14_jax_numpy_parity(self):
        """Test that JAX and numpy paths produce identical results."""
        try:
            import jax
            import jax.numpy as jnp
            jax.config.update("jax_enable_x64", True)
        except ImportError:
            self.skipTest("JAX not available")

        print("Test JAX/numpy parity.")
        start_cpu_time = time.process_time()

        THETA, spikes = generate_test_data(self.T, self.R, self.N)

        # --- Test compute_eta_G ---
        theta = np.random.RandomState(42).randn(self.N, self.N + 1)
        F1 = np.empty((self.R, self.N + 1))
        F1[:, 0] = 1.0
        F1[:, 1:] = spikes[0]

        # Numpy path
        r = 1 / (1 + np.exp(-F1 @ theta.T))
        eta_np = r.T @ F1
        w = r * (1 - r)
        G_np = np.empty((self.N, self.N + 1, self.N + 1))
        for n in range(self.N):
            wF = F1 * w[:, n:n+1]
            G_np[n] = wF.T @ F1

        # JAX path
        eta_jax, G_jax = exp_max._compute_eta_G_jax(
            jnp.asarray(theta), jnp.asarray(F1))
        eta_jax, G_jax = np.asarray(eta_jax), np.asarray(G_jax)

        eta_diff = np.max(np.abs(eta_np - eta_jax))
        G_diff = np.max(np.abs(G_np - G_jax))
        print('compute_eta_G: eta diff=%.2e, G diff=%.2e' % (eta_diff, G_diff))
        self.assertLess(eta_diff, 1e-10, "eta mismatch: %.2e" % eta_diff)
        self.assertLess(G_diff, 1e-10, "G mismatch: %.2e" % G_diff)

        # --- Test _compute_psi_jax ---
        # Run EM to get realistic theta_f
        emd = run_em_with_q_method(spikes, q_method='diagonal')

        # Numpy PSI (reference)
        bias = emd.theta_f[:, :, 0][:, np.newaxis, :]
        weights = emd.theta_f[:, :, 1:]
        logit = bias + np.matmul(emd.spikes[:emd.T],
                                 weights.swapaxes(-2, -1))
        PSI_np = np.sum(np.logaddexp(0, logit), axis=1)  # (T, N)

        # JAX PSI
        PSI_jax = np.empty((emd.T, emd.N))
        for t in range(emd.T):
            PSI_jax[t] = np.asarray(probability._compute_psi_jax(
                jnp.asarray(emd.spikes[t]),
                jnp.asarray(emd.theta_f[t])
            ))

        psi_diff = np.max(np.abs(PSI_np - PSI_jax))
        print('PSI: max diff=%.2e' % psi_diff)
        self.assertLess(psi_diff, 1e-10, "PSI mismatch: %.2e" % psi_diff)

        # --- Test full EM parity: force numpy path, compare mllk ---
        # The existing tests already check mllk against expected values,
        # so if JAX path produces the same mllk, parity is confirmed.
        print('EM mllk (JAX path): %.6f (expected %.6f)' %
              (emd.mllk, EXPECTED_MLLK_DIAGONAL_Q))
        self.assertFalse(
            np.absolute(emd.mllk - EXPECTED_MLLK_DIAGONAL_Q) > DEFAULT_MLLK_TOLERANCE,
            "JAX-path EM mllk should match expected value")

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))


    def test_15_jax_scan_parity(self):
        """Test that JAX scan E-step matches numpy fallback for all outputs."""
        try:
            import jax
        except ImportError:
            self.skipTest("JAX not available")

        print("Test JAX scan/numpy parity (full EM).")
        start_cpu_time = time.process_time()

        THETA, spikes = generate_test_data(self.T, self.R, self.N)

        # Run with JAX scan path (default when JAX available)
        emd_jax = run_em_with_q_method(spikes, q_method='diagonal')

        # Force numpy fallback and re-run
        saved_em = exp_max._HAS_JAX
        saved_pr = probability._HAS_JAX
        exp_max._HAS_JAX = False
        probability._HAS_JAX = False
        try:
            emd_np = run_em_with_q_method(spikes, q_method='diagonal')
        finally:
            exp_max._HAS_JAX = saved_em
            probability._HAS_JAX = saved_pr

        # Compare all state arrays
        fields = ['theta_f', 'sigma_f', 'theta_s', 'sigma_s',
                  'theta_o', 'sigma_o', 'sigma_o_i']
        for name in fields:
            arr_jax = getattr(emd_jax, name)
            arr_np = getattr(emd_np, name)
            diff = np.max(np.abs(arr_jax - arr_np))
            print('%s: max diff=%.2e' % (name, diff))
            self.assertLess(diff, 1e-8, "%s mismatch: %.2e" % (name, diff))

        if emd_jax.T > 1:
            A_diff = np.max(np.abs(emd_jax.A - emd_np.A))
            loc_diff = np.max(np.abs(
                emd_jax.lag_one_covariance[:emd_jax.T - 1] -
                emd_np.lag_one_covariance[:emd_np.T - 1]))
            print('A: max diff=%.2e' % A_diff)
            print('lag_one_covariance: max diff=%.2e' % loc_diff)
            self.assertLess(A_diff, 1e-8, "A mismatch: %.2e" % A_diff)
            self.assertLess(loc_diff, 1e-8,
                            "lag_one_cov mismatch: %.2e" % loc_diff)

        mllk_diff = abs(emd_jax.mllk - emd_np.mllk)
        print('mllk: JAX=%.6f, numpy=%.6f, diff=%.2e' %
              (emd_jax.mllk, emd_np.mllk, mllk_diff))
        self.assertLess(mllk_diff, 1e-6,
                        "mllk mismatch: %.2e" % mllk_diff)

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))


if __name__ == "__main__":
    unittest.main()
