"""
Minimal working example of the state-space kinetic Ising model, examining
time-varying field and coupling parameters between neurons. Note that modules
are imported at the top of each section that uses them (in contrast to the
usual convention of importing all modules at the top of the file) so as to
be completely explicit about the external requirements.

---

This code implements the state-space kinetic Ising model described in:
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


# Set time bins, number of trials, and number of neurons
T, R, N = 400, 200, 2


# ----- SPIKE SYNTHESIS -----
# Global module
import numpy as np
# Local module
import synthesis

# Set random seed for reproducibility
np.random.seed(42)
# Create underlying time-varying theta parameters as Gaussian processes
THETA = synthesis.get_THETA_gaussian_process(T, N, mu=1.0, sigma=10.0)
# Generate spike data from the kinetic Ising model
np.random.seed(1)
spikes = synthesis.get_S_function(T, R, N, THETA)


# ----- ALGORITHM EXECUTION -----
# Global module
import numpy as np
# Local module
import __init__  # From outside this folder, this would be 'import ssll_kinetic'

# Run the EM algorithm!
emd = __init__.run(spikes, max_iter=100, mstep=True)
# Without M-step (fixed state covariance):
#emd = __init__.run(spikes, max_iter=100, mstep=False)


# ----- ENTROPY FLOW -----
# Local module
import macro

# Compute entropy flow from estimated parameters
sf_bath, sr_bath, s_bath, M = macro.calculate_entropy_flow(emd)
# s_bath: net entropy flow (dissipative), shape (T, N)
# sf_bath: forward conditional entropy, shape (T, N)
# sr_bath: reverse conditional entropy, shape (T, N)
# M: mean-field spike probabilities, shape (T, N)


# ----- PLOTTING -----
# Global modules
import os
import matplotlib
matplotlib.use('Agg')
import pylab

# Create output directory
fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig')
os.makedirs(fig_dir, exist_ok=True)

# Set up an output figure
fig, ax = pylab.subplots(3, 1, sharex=True, figsize=(10, 8))

# Plot underlying theta traces (dashed) and estimated theta traces (solid)
# Field parameters (theta_{i,t})
for i in range(N):
    color = ['b', 'r'][i]
    ax[0].plot(THETA[:, i, 0], c=color, linestyle='--', alpha=0.7)
    ax[0].plot(emd.theta_s[:, i, 0], c=color)
ax[0].set_title('State-space kinetic Ising model (%d neurons)' % N)
ax[0].set_ylabel('Field parameters')
ax[0].legend(['True', 'Estimated'], loc='upper right')

# Coupling parameters (theta_{ij,t})
for i in range(N):
    for j in range(1, N + 1):
        color = ['g', 'm', 'c', 'y'][(i * N + j - 1) % 4]
        ax[1].plot(THETA[:, i, j], c=color, linestyle='--', alpha=0.7)
        ax[1].plot(emd.theta_s[:, i, j], c=color)
ax[1].set_ylabel('Coupling parameters')

# Entropy flow
entropy_flow = np.sum(s_bath, axis=1)  # Sum over neurons
mean_spikes = np.sum(M, axis=1)  # Sum over neurons
ax[2].plot(entropy_flow, c='k', label='Entropy flow')
ax[2].plot(mean_spikes, c='gray', linestyle='--', label='Mean spikes')
ax[2].set_xlabel('Time bins')
ax[2].set_ylabel('Entropy flow')
ax[2].legend(loc='upper right')

# Save figure
pylab.tight_layout()
fig_path = os.path.join(fig_dir, 'example_kinetic.png')
pylab.savefig(fig_path, dpi=150)
print('Figure saved to %s' % fig_path)
