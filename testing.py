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
DEFAULT_MLL_TOLERANCE = 1e-6  # Tolerance for log marginal likelihood comparison
DEFAULT_ENTROPY_TOLERANCE = 1e-4  # Tolerance for entropy flow comparison

# Expected Spike Count
EXPECTED_SPIKE_COUNT = 1094  # Total spikes for T=20, R=20, N=3

# Expected Log Marginal Likelihood Values
EXPECTED_MLL_DIAGONAL_Q = -261.296332
EXPECTED_MLL_FULL_Q = -256.209241
EXPECTED_MLL_SCALAR_Q = -263.798103
EXPECTED_MLL_NO_MSTEP = -278.074169

# Edge Case Expected Values
EXPECTED_MLL_SINGLE_NEURON = -142.615633
EXPECTED_MLL_SINGLE_TRIAL = -17.170574

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
    return ssll_kinetic.run(spikes, max_iter=max_iter, state_cov=state_cov,
                            EM_Info=False)


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
              (emd.mll, EXPECTED_MLL_DIAGONAL_Q))
        self.assertFalse(
            np.absolute(emd.mll - EXPECTED_MLL_DIAGONAL_Q) > DEFAULT_MLL_TOLERANCE,
            "Diagonal Q mll mismatch: got %.6f, expected %.6f" %
            (emd.mll, EXPECTED_MLL_DIAGONAL_Q))

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
              (emd.mll, EXPECTED_MLL_FULL_Q))
        self.assertFalse(
            np.absolute(emd.mll - EXPECTED_MLL_FULL_Q) > DEFAULT_MLL_TOLERANCE,
            "Full Q mll mismatch: got %.6f, expected %.6f" %
            (emd.mll, EXPECTED_MLL_FULL_Q))

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
              (emd.mll, EXPECTED_MLL_SCALAR_Q))
        self.assertFalse(
            np.absolute(emd.mll - EXPECTED_MLL_SCALAR_Q) > DEFAULT_MLL_TOLERANCE,
            "Scalar Q mll mismatch: got %.6f, expected %.6f" %
            (emd.mll, EXPECTED_MLL_SCALAR_Q))

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_4_no_mstep(self):
        """Test EM without M-step (E-step only, fixed state covariance)."""
        print("Test No M-step (T=%d, R=%d, N=%d)." %
              (self.T, self.R, self.N))
        start_cpu_time = time.process_time()

        THETA, spikes = generate_test_data(self.T, self.R, self.N)
        emd = ssll_kinetic.run(spikes, max_iter=100, mstep=False, EM_Info=False)

        print('Log marginal likelihood = %.6f (expected %.6f)' %
              (emd.mll, EXPECTED_MLL_NO_MSTEP))
        self.assertFalse(
            np.absolute(emd.mll - EXPECTED_MLL_NO_MSTEP) > DEFAULT_MLL_TOLERANCE,
            "No M-step mll mismatch: got %.6f, expected %.6f" %
            (emd.mll, EXPECTED_MLL_NO_MSTEP))

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_5_edge_case_single_neuron(self):
        """Test with a single neuron (N=1)."""
        print("Test Single Neuron (N=1).")
        start_cpu_time = time.process_time()

        THETA, spikes = generate_test_data(self.T, self.R, N=1)
        emd = run_em_with_q_method(spikes, q_method='diagonal')

        print('Log marginal likelihood = %.6f (expected %.6f)' %
              (emd.mll, EXPECTED_MLL_SINGLE_NEURON))
        self.assertFalse(
            np.absolute(emd.mll - EXPECTED_MLL_SINGLE_NEURON) > DEFAULT_MLL_TOLERANCE,
            "Single neuron mll mismatch: got %.6f, expected %.6f" %
            (emd.mll, EXPECTED_MLL_SINGLE_NEURON))

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_6_edge_case_single_trial(self):
        """Test with a single trial (R=1)."""
        print("Test Single Trial (R=1).")
        start_cpu_time = time.process_time()

        THETA, spikes = generate_test_data(self.T, R=1, N=self.N)
        emd = run_em_with_q_method(spikes, q_method='diagonal')

        print('Log marginal likelihood = %.6f (expected %.6f)' %
              (emd.mll, EXPECTED_MLL_SINGLE_TRIAL))
        self.assertFalse(
            np.absolute(emd.mll - EXPECTED_MLL_SINGLE_TRIAL) > DEFAULT_MLL_TOLERANCE,
            "Single trial mll mismatch: got %.6f, expected %.6f" %
            (emd.mll, EXPECTED_MLL_SINGLE_TRIAL))

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
        mll_array = np.array(emd.mll_list)
        diffs = np.diff(mll_array)
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

        emd = ssll_kinetic.run(spikes, max_iter=50, stationary=True,
                               EM_Info=False)

        # Should have pooled T*R trials into a single time step
        self.assertEqual(emd.T, 1)
        self.assertEqual(emd.R, T * R)
        self.assertEqual(emd.theta_s.shape, (1, N, N + 1))

        # Q should remain zero throughout
        self.assertTrue(np.all(emd.state_cov == 0),
            "State covariance Q should remain zero for stationary model")

        # dim_pram and AIC should be set
        self.assertEqual(emd.dim_param, N * (N + 1))
        expected_aic = -2 * emd.mll + 2 * emd.dim_param
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

        emd = ssll_kinetic.run(spikes, max_iter=50, stationary=True,
                               EM_Info=False)
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

        # --- Test full EM parity: force numpy path, compare mll ---
        # The existing tests already check mll against expected values,
        # so if JAX path produces the same mll, parity is confirmed.
        print('EM mll (JAX path): %.6f (expected %.6f)' %
              (emd.mll, EXPECTED_MLL_DIAGONAL_Q))
        self.assertFalse(
            np.absolute(emd.mll - EXPECTED_MLL_DIAGONAL_Q) > DEFAULT_MLL_TOLERANCE,
            "JAX-path EM mll should match expected value")

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

        mll_diff = abs(emd_jax.mll - emd_np.mll)
        print('mll: JAX=%.6f, numpy=%.6f, diff=%.2e' %
              (emd_jax.mll, emd_np.mll, mll_diff))
        self.assertLess(mll_diff, 1e-6,
                        "mll mismatch: %.2e" % mll_diff)

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))


    def test_16_exogenous_basic(self):
        """Test EM with exogenous input u: U shape, convergence, finite mll."""
        print("Test Exogenous Input (basic).")
        start_cpu_time = time.process_time()

        THETA, spikes = generate_test_data(self.T, self.R, self.N)
        d_u = 2
        np.random.seed(99)
        u = np.random.randn(self.T, d_u) * 0.1

        emd = ssll_kinetic.run(spikes, max_iter=500, state_cov=0.5, u=u,
                                EM_Info=False)

        # U should have correct shape
        self.assertEqual(emd.U.shape, (self.N, self.N + 1, d_u))
        # Should converge
        self.assertLess(emd.iterations, 500,
                        "EM with exogenous input should converge")
        # mll should be finite
        self.assertTrue(np.isfinite(emd.mll),
                        "mll should be finite with exogenous input")
        # AIC should include U params
        expected_u_params = self.N * (self.N + 1) * d_u
        # dim_param from scalar Q + U params
        self.assertGreaterEqual(emd.dim_param, expected_u_params)

        end_cpu_time = time.process_time()
        print('mll=%.6f, iterations=%d, U norm=%.4f' %
              (emd.mll, emd.iterations, np.linalg.norm(emd.U)))
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_17_exogenous_zero_u(self):
        """Test that u=zeros produces identical results to u=None."""
        print("Test Exogenous Zero Input (regression).")
        start_cpu_time = time.process_time()

        THETA, spikes = generate_test_data(self.T, self.R, self.N)
        d_u = 2
        u_zero = np.zeros((self.T, d_u))

        emd_none = ssll_kinetic.run(spikes, max_iter=50, state_cov=0.5,
                                     u=None, EM_Info=False)
        emd_zero = ssll_kinetic.run(spikes, max_iter=50, state_cov=0.5,
                                     u=u_zero, EM_Info=False)

        # theta_s should be identical (G*0 = 0 at every step)
        diff = np.max(np.abs(emd_none.theta_s - emd_zero.theta_s))
        print('theta_s max diff (None vs zeros): %.2e' % diff)
        self.assertLess(diff, 1e-8,
                        "u=zeros should match u=None: diff=%.2e" % diff)

        mll_diff = abs(emd_none.mll - emd_zero.mll)
        print('mll diff: %.2e' % mll_diff)
        self.assertLess(mll_diff, 1e-6,
                        "mll should match: diff=%.2e" % mll_diff)

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_18_exogenous_U_recovery(self):
        """Test that EM recovers the true U matrix from synthetic data.

        Generates theta trajectories with a known U and u via the state equation
        theta_t = theta_{t-1} + U*u_t + xi_t, then generates spikes from those
        thetas. The EM should recover U with high correlation to the true U.
        """
        print("Test Exogenous U Recovery.")
        start_cpu_time = time.process_time()

        T, R, N = 200, 200, 2
        d_u = 2
        rng = np.random.RandomState(42)

        # True U: (N, N+1, d_u) — moderate magnitude
        U_true = rng.randn(N, N + 1, d_u) * 0.3

        # Exogenous input: smooth sinusoidal signals
        t_axis = np.arange(T)
        u = np.column_stack([
            np.sin(2 * np.pi * t_axis / 50),
            np.cos(2 * np.pi * t_axis / 30),
        ])  # (T, d_u)

        # Generate theta trajectories: theta_t = theta_{t-1} + U*u_t + xi_t
        state_noise_std = 0.02
        THETA = np.zeros((T, N, N + 1))
        THETA[0] = rng.randn(N, N + 1) * 0.5
        for t in range(1, T):
            Uu = np.einsum('ndu,u->nd', U_true, u[t])
            THETA[t] = THETA[t - 1] + Uu + rng.randn(N, N + 1) * state_noise_std

        # Generate spikes from these thetas
        rng2 = np.random.RandomState(7)
        spikes = np.zeros((T + 1, R, N))
        rand_numbers = rng2.rand(T + 1, R, N)
        spikes[0] = (rand_numbers[0] >= 0.5).astype(int)
        for t in range(1, T + 1):
            F_psi = np.concatenate([np.ones((R, 1)), spikes[t - 1]], axis=1)
            logit = THETA[t - 1] @ F_psi.T  # (N, R)
            prob = 1 / (1 + np.exp(-logit))  # (N, R)
            spikes[t] = (prob.T >= rand_numbers[t]).astype(int)

        # Run EM with exogenous input
        emd = ssll_kinetic.run(spikes, max_iter=500, state_cov=0.5, u=u,
                                EM_Info=False)

        # Check U recovery: correlation between true and estimated U
        U_est = emd.U
        corr = np.corrcoef(U_true.ravel(), U_est.ravel())[0, 1]
        # Relative error
        rel_err = np.linalg.norm(U_est - U_true) / np.linalg.norm(U_true)

        print('U recovery: correlation=%.4f, relative error=%.4f' %
              (corr, rel_err))
        print('U_true (flat):', np.round(U_true.ravel(), 3))
        print('U_est  (flat):', np.round(U_est.ravel(), 3))

        self.assertGreater(corr, 0.9,
            "U recovery correlation should be > 0.9 (got %.4f)" % corr)
        self.assertLess(rel_err, 0.5,
            "U relative error should be < 0.5 (got %.4f)" % rel_err)

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_19_exogenous_jax_numpy_parity(self):
        """Test JAX and numpy paths agree with exogenous input."""
        try:
            import jax
        except ImportError:
            self.skipTest("JAX not available")

        print("Test Exogenous JAX/numpy parity.")
        start_cpu_time = time.process_time()

        THETA, spikes = generate_test_data(self.T, self.R, self.N)
        d_u = 2
        np.random.seed(99)
        u = np.random.randn(self.T, d_u) * 0.1

        # JAX path
        emd_jax = ssll_kinetic.run(spikes, max_iter=50, state_cov=0.5, u=u,
                                    EM_Info=False)

        # Force numpy fallback
        saved_em = exp_max._HAS_JAX
        saved_pr = probability._HAS_JAX
        exp_max._HAS_JAX = False
        probability._HAS_JAX = False
        try:
            emd_np = ssll_kinetic.run(spikes, max_iter=50, state_cov=0.5, u=u,
                                       EM_Info=False)
        finally:
            exp_max._HAS_JAX = saved_em
            probability._HAS_JAX = saved_pr

        for name in ['theta_f', 'sigma_f', 'theta_s', 'sigma_s']:
            diff = np.max(np.abs(getattr(emd_jax, name) - getattr(emd_np, name)))
            print('%s: max diff=%.2e' % (name, diff))
            self.assertLess(diff, 1e-8, "%s mismatch: %.2e" % (name, diff))

        mll_diff = abs(emd_jax.mll - emd_np.mll)
        print('mll diff: %.2e' % mll_diff)
        self.assertLess(mll_diff, 1e-6, "mll mismatch: %.2e" % mll_diff)

        # U should also agree
        U_diff = np.max(np.abs(emd_jax.U - emd_np.U))
        print('U diff: %.2e' % U_diff)
        self.assertLess(U_diff, 1e-8, "U mismatch: %.2e" % U_diff)

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))


    def test_20_obs_input_basic(self):
        """Test EM with observation input v: V shape, convergence, finite mll."""
        print("Test Observation Input (basic).")
        start_cpu_time = time.process_time()

        THETA, spikes = generate_test_data(self.T, self.R, self.N)
        d_v = 2
        np.random.seed(88)
        v = np.random.randn(self.T, d_v) * 0.1

        emd = ssll_kinetic.run(spikes, max_iter=500, state_cov=0.5, v=v,
                                EM_Info=False)

        # V should have correct shape
        self.assertEqual(emd.V.shape, (self.N, d_v))
        # Should converge
        self.assertLess(emd.iterations, 500,
                        "EM with observation input should converge")
        # mll should be finite
        self.assertTrue(np.isfinite(emd.mll),
                        "mll should be finite with observation input")
        # AIC should include V params
        expected_v_params = self.N * d_v
        self.assertGreaterEqual(emd.dim_param, expected_v_params)

        end_cpu_time = time.process_time()
        print('mll=%.6f, iterations=%d, V norm=%.4f' %
              (emd.mll, emd.iterations, np.linalg.norm(emd.V)))
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_21_obs_input_zero_v(self):
        """Test that v=zeros produces identical results to v=None."""
        print("Test Observation Zero Input (regression).")
        start_cpu_time = time.process_time()

        THETA, spikes = generate_test_data(self.T, self.R, self.N)
        d_v = 2
        v_zero = np.zeros((self.T, d_v))

        emd_none = ssll_kinetic.run(spikes, max_iter=50, state_cov=0.5,
                                     v=None, EM_Info=False)
        emd_zero = ssll_kinetic.run(spikes, max_iter=50, state_cov=0.5,
                                     v=v_zero, EM_Info=False)

        # theta_s should be identical (V*0 = 0 at every step)
        diff = np.max(np.abs(emd_none.theta_s - emd_zero.theta_s))
        print('theta_s max diff (None vs zeros): %.2e' % diff)
        self.assertLess(diff, 1e-8,
                        "v=zeros should match v=None: diff=%.2e" % diff)

        mll_diff = abs(emd_none.mll - emd_zero.mll)
        print('mll diff: %.2e' % mll_diff)
        self.assertLess(mll_diff, 1e-6,
                        "mll should match: diff=%.2e" % mll_diff)

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_22_obs_input_V_recovery(self):
        """Test that EM recovers the true V matrix from synthetic data.

        Generates spikes with known V and v added to the observation model,
        then runs EM to recover V.
        """
        print("Test Observation V Recovery.")
        start_cpu_time = time.process_time()

        T, R, N = 200, 200, 2
        d_v = 2
        rng = np.random.RandomState(42)

        # True V: (N, d_v)
        V_true = rng.randn(N, d_v) * 0.5

        # Observation input: smooth sinusoidal signals
        t_axis = np.arange(T)
        v = np.column_stack([
            np.sin(2 * np.pi * t_axis / 50),
            np.cos(2 * np.pi * t_axis / 30),
        ])  # (T, d_v)

        # Generate theta trajectories (no V influence on state)
        THETA = synthesis.generate_thetas_fixed_seed(T, N, mu=-2.0, sigma=50.0,
                                                      alpha=12.0, base_seed=100)

        # Generate spikes with V*v in observation model
        rng2 = np.random.RandomState(7)
        spikes = np.zeros((T + 1, R, N))
        rand_numbers = rng2.rand(T + 1, R, N)
        spikes[0] = (rand_numbers[0] >= 0.5).astype(int)
        for t in range(1, T + 1):
            F_psi = np.concatenate([np.ones((R, 1)), spikes[t - 1]], axis=1)
            logit = THETA[t - 1] @ F_psi.T  # (N, R)
            obs_offset = V_true @ v[t - 1]  # (N,)
            logit = logit + obs_offset[:, np.newaxis]
            prob = 1 / (1 + np.exp(-logit))  # (N, R)
            spikes[t] = (prob.T >= rand_numbers[t]).astype(int)

        # Run EM with observation input
        emd = ssll_kinetic.run(spikes, max_iter=500, state_cov=0.5, v=v,
                                EM_Info=False)

        # Check V recovery
        V_est = emd.V
        corr = np.corrcoef(V_true.ravel(), V_est.ravel())[0, 1]
        rel_err = np.linalg.norm(V_est - V_true) / np.linalg.norm(V_true)

        print('V recovery: correlation=%.4f, relative error=%.4f' %
              (corr, rel_err))
        print('V_true (flat):', np.round(V_true.ravel(), 3))
        print('V_est  (flat):', np.round(V_est.ravel(), 3))

        self.assertGreater(corr, 0.9,
            "V recovery correlation should be > 0.9 (got %.4f)" % corr)

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_23_obs_input_jax_numpy_parity(self):
        """Test JAX and numpy paths agree with observation input."""
        try:
            import jax
        except ImportError:
            self.skipTest("JAX not available")

        print("Test Observation JAX/numpy parity.")
        start_cpu_time = time.process_time()

        THETA, spikes = generate_test_data(self.T, self.R, self.N)
        d_v = 2
        np.random.seed(88)
        v = np.random.randn(self.T, d_v) * 0.1

        # JAX path
        emd_jax = ssll_kinetic.run(spikes, max_iter=50, state_cov=0.5, v=v,
                                    EM_Info=False)

        # Force numpy fallback
        saved_em = exp_max._HAS_JAX
        saved_pr = probability._HAS_JAX
        exp_max._HAS_JAX = False
        probability._HAS_JAX = False
        try:
            emd_np = ssll_kinetic.run(spikes, max_iter=50, state_cov=0.5, v=v,
                                       EM_Info=False)
        finally:
            exp_max._HAS_JAX = saved_em
            probability._HAS_JAX = saved_pr

        for name in ['theta_f', 'sigma_f', 'theta_s', 'sigma_s']:
            diff = np.max(np.abs(getattr(emd_jax, name) - getattr(emd_np, name)))
            print('%s: max diff=%.2e' % (name, diff))
            self.assertLess(diff, 1e-8, "%s mismatch: %.2e" % (name, diff))

        mll_diff = abs(emd_jax.mll - emd_np.mll)
        print('mll diff: %.2e' % mll_diff)
        self.assertLess(mll_diff, 1e-6, "mll mismatch: %.2e" % mll_diff)

        # V should also agree
        V_diff = np.max(np.abs(emd_jax.V - emd_np.V))
        print('V diff: %.2e' % V_diff)
        self.assertLess(V_diff, 1e-8, "V mismatch: %.2e" % V_diff)

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))

    def test_24_obs_and_state_input_combined(self):
        """Test EM with both u (state input) and v (observation input)."""
        print("Test Combined State + Observation Input.")
        start_cpu_time = time.process_time()

        THETA, spikes = generate_test_data(self.T, self.R, self.N)
        d_u = 2
        d_v = 2
        np.random.seed(99)
        u = np.random.randn(self.T, d_u) * 0.1
        np.random.seed(88)
        v = np.random.randn(self.T, d_v) * 0.1

        emd = ssll_kinetic.run(spikes, max_iter=500, state_cov=0.5, u=u, v=v,
                                EM_Info=False)

        # Both U and V should be present
        self.assertIsNotNone(emd.U)
        self.assertIsNotNone(emd.V)
        self.assertEqual(emd.U.shape, (self.N, self.N + 1, d_u))
        self.assertEqual(emd.V.shape, (self.N, d_v))

        # Should converge
        self.assertLess(emd.iterations, 500,
                        "EM with combined input should converge")
        # mll should be finite
        self.assertTrue(np.isfinite(emd.mll),
                        "mll should be finite with combined input")

        # AIC should include both U and V params
        expected_uv_params = (self.N * (self.N + 1) * d_u +
                              self.N * d_v)
        self.assertGreaterEqual(emd.dim_param, expected_uv_params)

        end_cpu_time = time.process_time()
        print('mll=%.6f, iterations=%d' % (emd.mll, emd.iterations))
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))


    def test_25_stationary_workaround_state_cov_zero_v(self):
        """Test state_cov=0 workaround for stationary model with time-varying v.

        With state_cov=0, theta is constrained to be constant (no state noise)
        while preserving per-time-step observation input v. This should recover
        V better than stationary=True which time-averages v.
        """
        print("Test state_cov=0 workaround with time-varying v.")
        start_cpu_time = time.process_time()

        T, R, N = 50, 100, 2
        d_v = 1
        rng = np.random.RandomState(42)

        # True V: (N, d_v)
        V_true = rng.randn(N, d_v) * 0.5

        # Sinusoidal observation input: (T, d_v)
        t_axis = np.arange(T)
        v = np.sin(2 * np.pi * t_axis / 25)[:, np.newaxis]  # (T, 1)

        # Constant theta (stationary ground truth)
        theta_const = np.array([[-2.0, 0.3, -0.2],
                                [-1.8, -0.1, 0.25]])  # (N, N+1)
        THETA = np.tile(theta_const, (T, 1, 1))  # (T, N, N+1)

        # Generate spikes with V*v in observation model
        rng2 = np.random.RandomState(7)
        spikes = np.zeros((T + 1, R, N))
        rand_numbers = rng2.rand(T + 1, R, N)
        spikes[0] = (rand_numbers[0] >= 0.5).astype(int)
        for t in range(1, T + 1):
            F_psi = np.concatenate([np.ones((R, 1)), spikes[t - 1]], axis=1)
            logit = THETA[t - 1] @ F_psi.T  # (N, R)
            obs_offset = V_true @ v[t - 1]  # (N,)
            logit = logit + obs_offset[:, np.newaxis]
            prob = 1 / (1 + np.exp(-logit))  # (N, R)
            spikes[t] = (prob.T >= rand_numbers[t]).astype(int)

        # Run with state_cov=0 workaround
        emd_workaround = ssll_kinetic.run(spikes, max_iter=500, state_cov=0,
                                           v=v, EM_Info=False)

        # 1. Theta should be approximately constant across time
        theta_s = emd_workaround.theta_s  # (T, N, N+1)
        theta_std = np.std(theta_s, axis=0)  # (N, N+1)
        print('Theta std across time (max): %.6f' % theta_std.max())
        self.assertLess(theta_std.max(), 0.1,
            "Theta should be ~constant with state_cov=0 (max std=%.4f)" %
            theta_std.max())

        # 2. V recovery: correlation with true V
        V_est = emd_workaround.V
        corr_workaround = np.corrcoef(V_true.ravel(), V_est.ravel())[0, 1]
        print('V recovery (workaround): corr=%.4f' % corr_workaround)
        print('V_true:', np.round(V_true.ravel(), 3))
        print('V_est :', np.round(V_est.ravel(), 3))
        self.assertGreater(corr_workaround, 0.9,
            "V recovery correlation should be > 0.9 (got %.4f)" %
            corr_workaround)

        # 3. EM converges, mll is finite
        self.assertTrue(np.isfinite(emd_workaround.mll),
                        "mll should be finite")
        self.assertLess(emd_workaround.iterations, 500,
                        "EM should converge within 500 iterations")

        # 4. Compare against stationary=True (which time-averages v)
        emd_stationary = ssll_kinetic.run(spikes, max_iter=500, stationary=True,
                                           v=v, EM_Info=False)
        V_stat = emd_stationary.V
        corr_stationary = np.corrcoef(V_true.ravel(), V_stat.ravel())[0, 1]
        # stationary=True time-averages a sinusoidal v to ~0, so V_stat may
        # be all-zero → corrcoef returns nan. Treat nan as failed recovery.
        if np.isnan(corr_stationary):
            corr_stationary = 0.0
        print('V recovery (stationary): corr=%.4f' % corr_stationary)
        print('V_stat:', np.round(V_stat.ravel(), 3))
        self.assertGreater(corr_workaround, corr_stationary,
            "state_cov=0 workaround (corr=%.4f) should recover V better "
            "than stationary=True (corr=%.4f)" %
            (corr_workaround, corr_stationary))

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))


    def test_26_trial_specific_v(self):
        """Test trial-specific observation input v with shape (T, R, d_v).

        Generates spikes with trial-specific V*v[t,r] in the observation model,
        then runs EM to recover V. Also verifies that shared v via 3D shape
        matches the 2D path.
        """
        print("Test Trial-Specific Observation Input v.")
        start_cpu_time = time.process_time()

        T, R, N = 100, 100, 2
        d_v = 2
        rng = np.random.RandomState(42)

        # True V: (N, d_v)
        V_true = rng.randn(N, d_v) * 0.5

        # Trial-specific observation input: (T, R, d_v)
        v_3d = rng.randn(T, R, d_v) * 0.5

        # Constant theta
        theta_const = np.array([[-2.0, 0.3, -0.2],
                                [-1.8, -0.1, 0.25]])
        THETA = np.tile(theta_const, (T, 1, 1))

        # Generate spikes with trial-specific V*v[t,r]
        rng2 = np.random.RandomState(7)
        spikes = np.zeros((T + 1, R, N))
        rand_numbers = rng2.rand(T + 1, R, N)
        spikes[0] = (rand_numbers[0] >= 0.5).astype(int)
        for t in range(1, T + 1):
            F_psi = np.concatenate([np.ones((R, 1)), spikes[t - 1]], axis=1)
            logit = THETA[t - 1] @ F_psi.T  # (N, R)
            obs_offset = v_3d[t - 1] @ V_true.T  # (R, N)
            logit = logit + obs_offset.T  # (N, R)
            prob = 1 / (1 + np.exp(-logit))
            spikes[t] = (prob.T >= rand_numbers[t]).astype(int)

        # Run EM with trial-specific v
        emd = ssll_kinetic.run(spikes, max_iter=500, state_cov=0,
                                v=v_3d, EM_Info=False)

        # V recovery
        V_est = emd.V
        corr = np.corrcoef(V_true.ravel(), V_est.ravel())[0, 1]
        print('V recovery (trial-specific v): corr=%.4f' % corr)
        print('V_true:', np.round(V_true.ravel(), 3))
        print('V_est :', np.round(V_est.ravel(), 3))
        self.assertGreater(corr, 0.9,
            "V recovery with trial-specific v should be > 0.9 (got %.4f)" %
            corr)

        # Verify backward compat: shared v via 2D and 3D should match
        THETA2, spikes2 = generate_test_data(self.T, self.R, self.N)
        np.random.seed(88)
        v_2d = np.random.randn(self.T, d_v) * 0.1
        v_3d_shared = np.broadcast_to(v_2d[:, np.newaxis, :],
                                       (self.T, self.R, d_v)).copy()

        emd_2d = ssll_kinetic.run(spikes2, max_iter=50, state_cov=0.5,
                                   v=v_2d, EM_Info=False)
        emd_3d = ssll_kinetic.run(spikes2, max_iter=50, state_cov=0.5,
                                   v=v_3d_shared, EM_Info=False)

        theta_diff = np.max(np.abs(emd_2d.theta_s - emd_3d.theta_s))
        mll_diff = abs(emd_2d.mll - emd_3d.mll)
        V_diff = np.max(np.abs(emd_2d.V - emd_3d.V))
        print('2D vs 3D shared v: theta diff=%.2e, mll diff=%.2e, V diff=%.2e'
              % (theta_diff, mll_diff, V_diff))
        self.assertLess(theta_diff, 1e-8,
                        "2D and 3D shared v should give same theta")
        self.assertLess(mll_diff, 1e-6,
                        "2D and 3D shared v should give same mll")
        self.assertLess(V_diff, 1e-8,
                        "2D and 3D shared v should give same V")

        self.assertTrue(np.isfinite(emd.mll))
        self.assertLess(emd.iterations, 500)

        end_cpu_time = time.process_time()
        print('Total CPU time: %.3f seconds' % (end_cpu_time - start_cpu_time))


if __name__ == "__main__":
    unittest.main()
