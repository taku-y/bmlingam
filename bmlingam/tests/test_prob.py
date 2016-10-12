# -*- coding: utf-8 -*-

"""Test functions for probability distributions.
"""
# Author: Taku Yoshioka
# License: MIT

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
from scipy import stats
from scipy.special import gamma

from bmlingam.prob import ll_laplace, ll_gg, sample_gg

def test_laplace_gg(plot=False):
    """Check if the outputs of ll_laplace() and ll_gg(, beta=0.5).

    Outputs should be equivalent up to numerical error.
    """
    xs = np.arange(-10., 10., .2)
    out1 = ll_laplace(xs)
    out2 = ll_gg(xs, beta=.5)

    if plot:
        plt.plot(xs, out1, 'b')
        plt.plot(xs, out2, 'g')

def _describe_and_check(txt, xs, ss):
    d = stats.describe(xs)
    print(txt)
    print('Mean: {}'.format(d.mean))
    print('Var : {}'.format(d.variance))
    print('Skew: {}'.format(d.skewness))
    print('Kurt: {}'.format(d.kurtosis))

    assert_allclose([d.mean, d.variance, d.skewness, d.kurtosis], 
                    ss, rtol=5e-2, atol=5e-2)

def _mv_kurtosis(xs):
    dim = xs.shape[1]
    prec = np.linalg.pinv(np.cov(xs.T))
    xs_ = xs - xs.mean(axis=0)
    print(xs_.shape, prec.shape)
    xpx = np.sum((xs_.dot(prec)) * xs_, axis=1)
    k = np.mean(xpx**2) - dim * (dim + 2)

    print('Mv kurtosis: {}'.format(k))

    return k

def test_sample_gg(n_samples=1000000, plot=False):
    """Tests for generalized Gaussian. 
    """
    rng = np.random.RandomState(0)

    # Test 1
    print('Test1')
    dim = 2
    scov = np.eye(dim)
    beta = 1.0
    xs = sample_gg(scov, beta, n_samples, rng, dim, normalize=True)
    _describe_and_check('xs[:, 0]', xs[:, 0], [0, 1, 0, 0])
    _describe_and_check('xs[:, 1]', xs[:, 1], [0, 1, 0, 0])

    # Test 2
    print('\nTest2')
    dim = 2
    scov = np.array([[1.0, 0.5], [0.5, 1.0]])
    beta = 1.0
    xs = sample_gg(scov, beta, n_samples, rng, dim, normalize=True)
    _describe_and_check('xs[:, 0]', xs[:, 0], [0, 1, 0, 0])
    _describe_and_check('xs[:, 1]', xs[:, 1], [0, 1, 0, 0])

    # Test 3
    print('\nTest3')
    dim = 1
    scov = np.eye(dim)
    beta = 0.5
    xs = sample_gg(scov, beta, n_samples, rng, dim, normalize=True)
    _describe_and_check('xs', xs.ravel(), [0, 1, 0, 3])

    # Test 4
    print('\nTest4')
    dim = 2
    scov = np.eye(dim)
    beta = 0.5
    xs = sample_gg(scov, beta, n_samples, rng, dim, normalize=True)
    k = _mv_kurtosis(xs)
    k_true = ((dim**2) * gamma(dim / (2 * beta)) * gamma((dim + 4) / (2 * beta))) / \
             (gamma((dim + 2) / (2 * beta))**2) - dim * (dim + 2)
    print('True: {}\n'.format(k_true))
    assert_allclose(k, k_true, atol=5e-2, rtol=5e-2)
