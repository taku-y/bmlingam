# -*- coding: utf-8 -*-

"""Implementation of probability distributions.
"""
# Author: Taku Yoshioka, Shohei Shimizu
# License: MIT

import numpy as np
from numpy import log, sqrt, abs, power, sign
from scipy.special import gamma, gammaln
from pymc3.distributions import Continuous
from theano import tensor as tt
import theano
from theano.tensor.nlinalg import pinv
from theano.gof import Op, Apply

# The code is adopted from https://github.com/Theano/Theano/pull/3959
class LogAbsDet(Op):
    """Computes the logarithm of absolute determinant of a square
    matrix M, log(abs(det(M))), on CPU. Avoids det(M) overflow/
    underflow.
    TODO: add GPU code!
    """
    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        o = theano.tensor.scalar(dtype=x.dtype)
        return Apply(self, [x], [o])

    def perform(self, node, inputs, outputs):
        try:
            (x,) = inputs
            (z,) = outputs
            s = np.linalg.svd(x, compute_uv=False)
            log_abs_det = np.sum(np.log(np.abs(s)))
            z[0] = np.asarray(log_abs_det, dtype=x.dtype)
        except Exception:
            print('Failed to compute logabsdet of {}.'.format(x))
            raise

    def grad(self, inputs, g_outputs):
        gz, = g_outputs
        x, = inputs
        # return [gz * tt.nlinalg.matrix_inverse(x).T]
        return [gz * pinv(x).T]

    def __str__(self):
        return "LogAbsDet"

logabsdet = LogAbsDet()

def ll_laplace(e):
    """Return of log likelihood of the standard Laplace distribution. 

    :param e: Sample values.
    :type e: ndarray, shape=(n_theta_sampling, n_samples)
    :return: pdf values at given samples. 
    :rtype: ndarray, shape=(n_theta_sampling, n_samples)
    """
    b = 1 / sqrt(2)
    return -log(2 * b) - abs(e) / b

def ll_gg(e, beta):
    """Return of log likelihood of generalized Gaussian distributions. 
    """
    beta = float(beta)
    m = gamma(0.5 / beta) / ((2**(1 / beta)) * gamma(3 / (2 * beta)))
    
    return - 0.5 * power((e**2) / m, beta) + log(beta) \
           - gammaln(0.5 / beta) - (0.5 / beta) * log(2) - 0.5 * log(m)

def sample_gg(scov, beta, n_samples, rng, dim, normalize):
    """Draw samples from GG distribution. 

    See: Gómez, E., Gomez-Viilegas, M. A., & Marin, J. M. (1998). 
    A multivariate generalization of the power exponential family of distributions. 
    Communications in Statistics-Theory and Methods, 27(3), 589-600.

    Parameters
    ----------
    scov: numpy.ndarray, shape=(dim, dim)
        A scaled covariance matrix. For beta=1, it becomes the covariance matrix. 

    beta: float
        Shape parameter, a positive value. 

    n_samples: int
        Number of samples. 

    rng: numpy.random.RandomState
        Random number generator. 

    dim: int 
        The dimension of the distribution. 

    normalize: bool
        If true, each dimension is normalized such that its variance is 1. 
    """
    # Draw samples from unit sphere
    gs = rng.normal(0, 1, size=(n_samples, dim))
    ss = gs + sign(sign(gs) + 1e-10) * 1e-10
    ns = np.linalg.norm(ss, ord=2, axis=1)[:, np.newaxis]
    us = ss / ns

    # Draw samples from GG
    S = np.linalg.cholesky(scov).T
    ts = rng.gamma(shape=dim / (2.0 * beta), scale=2.0, 
                   size=n_samples)[:, np.newaxis]
    xs = (ts**(1.0 / (2.0 * beta))) * us.dot(S)

    if normalize:
        v = (2**(1.0 / beta) * gamma((dim + 2.0) / (2.0 * beta))) / \
            (dim * gamma(dim / (2.0 * beta)))
        xs = xs / np.sqrt(v * np.diag(scov))

    return xs

class GeneralizedGaussian(Continuous):
    def __init__(self, mu=0.0, beta=None, cov=None, *args, **kwargs):
        super(GeneralizedGaussian, self).__init__(*args, **kwargs)
        # assert(mu.shape[0] == cov.shape[0] == cov.shape[1])
        dim = mu.shape[0]

        self.mu = mu
        self.beta = beta
        self.prec = tt.nlinalg.pinv(cov)
        # self.k = (dim * tt.gamma(dim / 2.0)) / \
        #          ((np.pi**(dim / 2.0)) * tt.gamma(1 + dim / (2 * beta)) * (2**(1 + dim / (2 * beta))))
        self.logk = tt.log(dim) + tt.gammaln(dim / 2.0) - \
                    (dim / 2.0) * tt.log(np.pi) - \
                    tt.gammaln(1 + dim / (2 * beta)) - \
                    (1 + dim / (2 * beta)) * tt.log(2.0)

    def logp(self, value):
        x = value - self.mu

        if x.tag.test_value.ndim == 1:
            xpx = tt.sum(x * self.prec * x)
            normalize = tt.log(self.prec)
        else:
            (x.dot(self.prec) * x).sum(axis=x.ndim - 1)
            # normalize = tt.log(tt.nlinalg.Det(self.prec))
            normalize = logabsdet(self.prec)

        # return tt.log(self.k) + 0.5 * normalize - 0.5 * xpx
        return self.logk + 0.5 * normalize - 0.5 * xpx

def multivariatet(mu, Sigma, N, M, rng):
    """Return a sample (or samples) from the multivariate t distribution.

    This function is adopted from 
    http://kennychowdhary.me/2013/03/python-code-to-generate-samples-from-multivariate-t/.

    :param mu: Mean.
    :type mu: ndarray, shape=(n_dim,), dtype=float
    :param Sigma: Scaling matrix.
    :type Sigma: ndarray, shape=(n_dim, n_dim), dtype=float
    :param float N: Degrees of freedom.
    :param int M: Number of samples to produce. 
    :param np.random.RandomState rng: Random number generator. 
    :return: M samples of (n_dim)-dimensional multivariate t distribution.
    :rtype: ndarray, shape=(n_samples, n_dim), dtype=float

    """
    d = len(Sigma)
    g = np.tile(rng.gamma(N/2., 2./N, M), (d, 1)).T
    Z = rng.multivariate_normal(np.zeros(d), Sigma, M)
    return mu + Z / np.sqrt(g)
