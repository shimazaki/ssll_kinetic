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
    return np.sum(f(x, *args)) * dx

################################################################
# Integrand functions
################################################################
def dT_s(x, g, D):
    """Integrand for forward entropy: Gaussian-weighted chi(g + x*sqrt(D))."""
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2) * chi(g + x * np.sqrt(D))


def dT1(x, g, D):
    """Integrand for mean spike update: Gaussian-weighted sigmoid(g + x*sqrt(D))."""
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2) * sigmoid(g + x * np.sqrt(D))


def dT_sr_0(x, g, D):
    """Reverse-entropy integrand for spike=0."""
    A = 0.0
    B = -np.log(1 + np.exp(g + x * np.sqrt(D)))
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2) * (A + B)


def dT_sr_1(x, g, D):
    """Reverse-entropy integrand for spike=1."""
    A = (g + x * np.sqrt(D))
    B = -np.log(1 + np.exp(g + x * np.sqrt(D)))
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2) * (A + B)

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
def computation_m(a, m_p):
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


def Dissipation_en(a, m, m_p):
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


def calculate_entropy_flow(emd):
    """
    Computes entropy flow time series from an EMData object after EM convergence.

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
    for t in range(emd.T - 1):
        m_p = mp if t == 0 else m
        THETA_st = emd.theta_s[t]
        m = computation_m(THETA_st, m_p)
        sf_bath_t, sr_bath_t, s_bath_t = Dissipation_en(THETA_st, m, m_p)
        sf_bath[t, :] = sf_bath_t
        sr_bath[t, :] = sr_bath_t
        s_bath[t, :] = s_bath_t
        M[t] = m
    return sf_bath, sr_bath, s_bath, M
