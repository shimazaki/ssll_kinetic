"""
This code implements the state-space kinetic Ising model described in:
Ken Ishihara, Hideaki Shimazaki. *State-space kinetic Ising model reveals task-dependent entropy flow in sparsely active nonequilibrium neuronal dynamics*. (2025) arXiv:2502.15440

The implementation extends existing libraries available at:
- https://github.com/christiando/ssll_lib.git
- https://github.com/shimazaki/dynamic_corr

This implementation also incorporates and adapts mean-field approximation techniques based on:
- https://github.com/MiguelAguilera/kinetic-Plefka-expansions.git

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

################################################################
# Sigmoid and entropy-related utilities
################################################################
def sigmoid(a):
    """Standard sigmoid function: sigmoid(a) = 1 / (1 + exp(-a))."""
    return 1 / (1 + np.exp(-a))


def chi(a):
    """
    Binary entropy function: chi(a) = -[ p*log(p) + (1 - p)*log(1 - p) ],
    where p = 1 / (1 + exp(a)).
    """
    oe = 1 / (1 + np.exp(a))
    return -(oe * np.log(oe) + (1 - oe) * np.log(1 - oe))

################################################################
# 1D Gaussian integration
################################################################
def integrate_1DGaussian(f, args=(), Nint=100):
    """
    1D Gaussian numerical integration over [-4, 4].

    Parameters
    ----------
    f : callable
        Function f(x, *args) to integrate.
    args : tuple
        Additional arguments for f.
    Nint : int
        Number of integration points.

    Returns
    -------
    float
        Approximate integral.
    """
    x = np.linspace(-1, 1, Nint) * 4
    dx = x[1] - x[0]
    w = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
    return np.sum(w * f(x, *args)) * dx

################################################################
# Integrand functions
################################################################
def dT_s(x, g, D):
    """Integrand for forward entropy: chi(g + x*sqrt(D)) (Eq. 48)."""
    return chi(g + x * np.sqrt(D))


def dT1(x, g, D):
    """Integrand for mean spike update: sigmoid(g + x*sqrt(D)) (Eq. 45)."""
    return sigmoid(g + x * np.sqrt(D))


def dT_sr_0(x, g, D):
    """Reverse-entropy integrand for spike=0: -log(1 + exp(h)) (Eq. 58)."""
    return -np.log(1 + np.exp(g + x * np.sqrt(D)))


def dT_sr_1(x, g, D):
    """Reverse-entropy integrand for spike=1: h - log(1 + exp(h)) (Eq. 58)."""
    h = g + x * np.sqrt(D)
    return h - np.log(1 + np.exp(h))

def dT_sr_h(x, g, D):
    """Integrand for expected natural parameter: h = g + x*sqrt(D).
    Computes E[h] where h = g + x*sqrt(D). Used in reverse entropy (Eq. 57)."""
    return g + x * np.sqrt(D)


def dT_sr_rh(x, g, D):
    """Integrand for E[r(h)*h]: r(h)*h where r(h) = sigmoid(h).
    Used in forward entropy h-psi decomposition (Eq. 47): chi(h) = -[r(h)*h - psi(h)]."""
    h = g + x * np.sqrt(D)
    return sigmoid(h) * h


def dT_sr_psi(x, g, D):
    """Integrand for expected log-partition: log(1 + exp(h)).
    Computes E[psi(h)] where psi(h) = log(1 + exp(h)) (Eq. 47)."""
    return np.log(1 + np.exp(g + x * np.sqrt(D)))

################################################################
# Forward/reverse entropy and mean spike updates
################################################################
def update_S(H, J, m_p):
    """
    Computes forward conditional entropy for each neuron.

    Parameters
    ----------
    H : np.ndarray, shape (N,)
        External fields.
    J : np.ndarray, shape (N, N)
        Coupling matrix.
    m_p : np.ndarray, shape (N,)
        Previous mean spikes.

    Returns
    -------
    S : np.ndarray, shape (N,)
        Forward entropy per neuron.
    """
    size = len(H)
    S = np.zeros(size)
    g = H + np.dot(J, m_p)
    D = np.dot(J**2, m_p * (1 - m_p))
    for i in range(size):
        S[i] = integrate_1DGaussian(dT_s, (g[i], D[i]))
    return S


def update_S_re(H, J, m, m_p):
    """
    Computes reverse conditional entropy for each neuron.

    Parameters
    ----------
    H : np.ndarray, shape (N,)
        External fields.
    J : np.ndarray, shape (N, N)
        Coupling matrix.
    m : np.ndarray, shape (N,)
        Current mean spikes.
    m_p : np.ndarray, shape (N,)
        Previous mean spikes.

    Returns
    -------
    S : np.ndarray, shape (N,)
        Reverse entropy per neuron.
    """
    size = len(H)
    phi_0 = np.zeros(size)
    phi_1 = np.zeros(size)
    S = np.zeros(size)
    g = H + np.dot(J, m)
    D = np.dot(J**2, m * (1 - m))
    for i in range(size):
        phi_0[i] = integrate_1DGaussian(dT_sr_0, (g[i], D[i]))
        phi_1[i] = integrate_1DGaussian(dT_sr_1, (g[i], D[i]))
        S[i] = -(m_p[i] * phi_1[i] + (1 - m_p[i]) * phi_0[i])
    return S


################################################################
# Alternative entropy formulation via h-psi decomposition
# These compute the same quantities as update_S / update_S_re
# but decompose conditional entropy as -(m*<h> - <psi>),
# corresponding to Eq. 47 in the paper:
#   chi(h) = -[r(h)*h - psi(h)]
################################################################
def update_S_alt(H, J, m, m_p):
    """
    Forward conditional entropy via h-psi decomposition (Eq. 47-48).

    Uses the identity chi(h) = -[r(h)*h - psi(h)] (Eq. 47) to compute:
      S[i] = -(<r(h_i)*h_i> - <psi(h_i)>)
    where expectations are Gaussian-integrated over the mean-field
    distribution with mean g = H + J*m_p and variance D = J^2 * m_p*(1-m_p).

    Equivalent to update_S. The decomposition separates the contribution
    of the expected natural parameter (weighted by firing rate) from the
    log-partition function.

    Parameters
    ----------
    H : np.ndarray, shape (N,)
        External fields.
    J : np.ndarray, shape (N, N)
        Coupling matrix.
    m : np.ndarray, shape (N,)
        Current mean spikes (unused in exact formulation, kept for API consistency).
    m_p : np.ndarray, shape (N,)
        Previous mean spikes.

    Returns
    -------
    S : np.ndarray, shape (N,)
        Forward entropy per neuron.
    """
    size = len(H)
    S = np.zeros(size)
    rh_vals = np.zeros(size)
    psi_vals = np.zeros(size)
    g = H + np.dot(J, m_p)
    D = np.dot(J**2, m_p * (1 - m_p))
    for i in range(size):
        rh_vals[i] = integrate_1DGaussian(dT_sr_rh, (g[i], D[i]))
        psi_vals[i] = integrate_1DGaussian(dT_sr_psi, (g[i], D[i]))
        S[i] = -(rh_vals[i] - psi_vals[i])
    return S


def update_S_re_alt(H, J, m, m_p):
    """
    Reverse conditional entropy via h-psi decomposition (Eq. 57-58).

    Computes: S[i] = -(m_p[i] * <h_i> - <psi_i>)
    where <h_i> and <psi_i> are Gaussian-integrated over the
    mean-field distribution with mean g = H + J*m (current spikes)
    and variance D = J^2 * m*(1-m).

    Equivalent to update_S_re but uses the h-psi decomposition
    with reversed roles of m and m_p.

    Parameters
    ----------
    H : np.ndarray, shape (N,)
        External fields.
    J : np.ndarray, shape (N, N)
        Coupling matrix.
    m : np.ndarray, shape (N,)
        Current mean spikes.
    m_p : np.ndarray, shape (N,)
        Previous mean spikes.

    Returns
    -------
    S : np.ndarray, shape (N,)
        Reverse entropy per neuron.
    """
    size = len(H)
    h_vals = np.zeros(size)
    psi_vals = np.zeros(size)
    S = np.zeros(size)
    g = H + np.dot(J, m)
    D = np.dot(J**2, m * (1 - m))
    for i in range(size):
        h_vals[i] = integrate_1DGaussian(dT_sr_h, (g[i], D[i]))
        psi_vals[i] = integrate_1DGaussian(dT_sr_psi, (g[i], D[i]))
        S[i] = -(m_p[i] * h_vals[i] - psi_vals[i])
    return S


def update_m_P_t1_o1(H, J, m_p):
    """
    Mean-field spike probability update.

    Parameters
    ----------
    H : np.ndarray, shape (N,)
    J : np.ndarray, shape (N, N)
    m_p : np.ndarray, shape (N,)
        Previous mean spikes.

    Returns
    -------
    m : np.ndarray, shape (N,)
        Updated mean spikes.
    """
    size = len(H)
    m = np.zeros(size)
    g = H + np.dot(J, m_p)
    D = np.dot(J**2, m_p * (1 - m_p))
    for i in range(size):
        m[i] = integrate_1DGaussian(dT1, (g[i], D[i]))
    return m

################################################################
# High-level functions
################################################################
def compute_mean_field(a, m_p):
    """
    Computes mean-field spike probabilities from parameter array a and
    previous mean spikes m_p.

    Parameters
    ----------
    a : np.ndarray, shape (N, N+1)
        Parameter array: a[:, 0] = external field H, a[:, 1:] = coupling J.
    m_p : np.ndarray, shape (N,)
        Previous mean spikes.

    Returns
    -------
    m : np.ndarray, shape (N,)
        Updated mean spikes.
    """
    H = a[:, 0]
    J = np.delete(a, 0, 1)
    m = update_m_P_t1_o1(H, J, m_p)
    return m


def compute_dissipation(a, m, m_p):
    """
    Computes forward and reverse conditional entropy and net entropy flow
    (dissipative entropy flow) per neuron.

    Parameters
    ----------
    a : np.ndarray, shape (N, N+1)
        Parameter array: a[:, 0] = external field H, a[:, 1:] = coupling J.
    m : np.ndarray, shape (N,)
        Current mean spikes.
    m_p : np.ndarray, shape (N,)
        Previous mean spikes.

    Returns
    -------
    S_forward : np.ndarray, shape (N,)
        Forward conditional entropy per neuron.
    S_reverse : np.ndarray, shape (N,)
        Reverse conditional entropy per neuron.
    net_flow : np.ndarray, shape (N,)
        Net entropy flow per neuron: -(S_forward - S_reverse).
    """
    H = a[:, 0]
    J = np.delete(a, 0, 1)
    S_forward = update_S(H, J, m_p)
    S_reverse = update_S_re(H, J, m, m_p)
    net_flow = -(S_forward - S_reverse)
    return S_forward, S_reverse, net_flow


def compute_entropy_flow(emd):
    """
    Computes entropy flow time series from an EMData object after EM convergence.

    For stationary models (emd.T == 1), iterates the mean-field equation
    m = f(theta, m) from the empirical spike mean to the fixed point m*,
    then computes entropy flow at (theta, m*, m*).

    Parameters
    ----------
    emd : container.EMData
        EMData object with smoothed parameters (theta_s) and spike data.

    Returns
    -------
    sf_bath : np.ndarray, shape (T, N)
        Forward conditional entropy per neuron per time step.
    sr_bath : np.ndarray, shape (T, N)
        Reverse conditional entropy per neuron per time step.
    s_bath : np.ndarray, shape (T, N)
        Net entropy flow (dissipative) per neuron per time step.
    M : np.ndarray, shape (T, N)
        Mean-field spike probabilities per neuron per time step.
    """
    M = np.zeros((emd.T, emd.N))
    s_bath = np.zeros((emd.T, emd.N))
    sf_bath = np.zeros((emd.T, emd.N))
    sr_bath = np.zeros((emd.T, emd.N))
    mp = np.mean(emd.spikes, axis=(0, 1))

    if emd.T == 1:
        # Stationary case: iterate mean-field to fixed point m*
        theta = emd.theta_s[0].copy()
        if emd.V is not None:
            theta[:, 0] = theta[:, 0] + emd.V @ emd.v[0].mean(axis=0)
        m = mp.copy()
        for _ in range(1000):
            m_prev = m.copy()
            m = compute_mean_field(theta, m)
            if np.max(np.abs(m - m_prev)) < 1e-8:
                break
        # At stationarity, m_p = m = m*
        sf, sr, s_net = compute_dissipation(theta, m, m)
        sf_bath[0, :] = sf
        sr_bath[0, :] = sr
        s_bath[0, :] = s_net
        M[0] = m
    else:
        for t in range(emd.T):
            m_p = mp if t == 0 else m
            THETA_st = emd.theta_s[t].copy()
            if emd.V is not None:
                THETA_st[:, 0] = THETA_st[:, 0] + emd.V @ emd.v[t].mean(axis=0)
            m = compute_mean_field(THETA_st, m_p)
            sf_bath_t, sr_bath_t, s_bath_t = compute_dissipation(THETA_st, m, m_p)
            sf_bath[t, :] = sf_bath_t
            sr_bath[t, :] = sr_bath_t
            s_bath[t, :] = s_bath_t
            M[t] = m

    return sf_bath, sr_bath, s_bath, M
