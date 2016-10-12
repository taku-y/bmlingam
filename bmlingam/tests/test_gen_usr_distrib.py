# -*- coding: utf-8 -*-

# Author: Taku Yoshioka, Shohei Shimizu
# License: MIT

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import skew, kurtosis

from bmlingam.tests.gendata import _gen_usr_distrib

def test_gen_usr_distrib(n_samples=100000, verbose=False):
    rng  = np.random.RandomState(0)

    xs = _gen_usr_distrib(n_samples, ['laplace'], rng)
    assert_allclose(np.mean(xs), 0, atol=5e-2)
    assert_allclose(np.std(xs), 1, atol=5e-2)
    assert_allclose(skew(xs)[0], 0, atol=5e-2)
    assert_allclose(kurtosis(xs)[0], 3, atol=5e-2)

    xs = _gen_usr_distrib(n_samples, ['exp'], rng)
    assert_allclose(np.std(xs), 1, atol=5e-2)
