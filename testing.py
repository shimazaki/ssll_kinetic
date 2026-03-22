import unittest

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
import synthesis
import __init__



class TestEstimator(unittest.TestCase):


    def test_1_fo_varying(self):
        T = 10
        R = 10
        N = 2

        np.random.seed(1)
        THETA = synthesis.get_THETA_gaussian_process(T, N)
        spikes = synthesis.get_S_function(T, R, N, THETA)

        print("Test First-Order Time-Varying Interactions.")

        emd = __init__.run(spikes, max_iter=100, mstep=True)

        # Check the consistency with the expected result.
        print('Log marginal likelihood = %.6f' % emd.mllk)


if __name__ == "__main__":
    unittest.main()
