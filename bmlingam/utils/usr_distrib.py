# -*- coding: utf-8 -*-

"""Distributions used in the Kernel ICA paper of Bach and Jordan. 

The program originally has been written by R. Bach as a part of
his kernel ICA software of MATLAB. 
"""
# Copyright (c) Francis R. Bach, 2002.
# Author: Taku Yoshioka

import collections
import numpy as np
import scipy.stats as stats

dist_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
              'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's']

def usr_distrib(dist_name, query, param=None, rng=None):
    """Return random numbers, pdf, name or kurtosis of distributions. 

    Parameter :code:`query` is either
    
    - 'rnd', param is then the number of samples
    - 'pdf', param is then a row of abcissas
    - 'name', param is then optional
    
    :param str dist_name: A letter between 'a' and 'r'.
    :param str query: 'rnd', 'pdf', 'name' or pdf.
    :param param: Depends on query. 
    :param np.random.RandomState rng: Random number generator. 

    """
    if query == 'pdf':
        param_ = wrap_array(param)

    # ---- mixture of 4 Gaussians, symmetric and multimodal ----
    if dist_name == 'm':
        props = np.array([1., 2., 2., 1.])
        props = props / np.sum(props)
        mus = np.array([-1, -.33, .33, 1])
        covs = np.array([.16, .16, .16, .16])

        if query == 'name':
            return 'Mix4Gauss_SymMultiModal'
        elif query == 'pdf':
            return _pdf_mixnorm(props, mus, covs, xs=param_)
        elif query == 'rnd':
            return _rnd_mixnorm(props, mus, covs, rng, n_samples=param)
        elif query == 'kurt':
            return _krt_mixnorm(props, mus, covs)
        else:
            raise ValueError('Invalid value of query: %s' % query)

    # ---- mixture of 4 Gaussians, symmetric and transitional ----
    elif dist_name == 'n':
        props = np.array([1., 2., 2., 1.])
        props = props / np.sum(props)
        mus = np.array([-1, -.2, .2, 1])
        covs = np.array([.2, .3, .3, .2])

        if query == 'name':
            return 'Mix4Gauss_SymTransitional'
        elif query == 'pdf':
            return _pdf_mixnorm(props, mus, covs, xs=param_)
        elif query == 'rnd':
            return _rnd_mixnorm(props, mus, covs, rng, n_samples=param)
        elif query == 'kurt':
            return _krt_mixnorm(props, mus, covs)
        else:
            raise ValueError('Invalid value of query: %s' % query)

    # ---- mixture of 4 Gaussians, symmetric and unimodal ----
    elif dist_name == 'o':
        props = np.array([1., 2., 2., 1.])
        props = props / np.sum(props)
        mus = np.array([-.7, -.2, .2, 7])
        covs = np.array([.2, .3, .3, .2])

        if query == 'name':
            return 'Mix4Gauss_SymUniModal'
        elif query == 'pdf':
            return _pdf_mixnorm(props, mus, covs, xs=param_)
        elif query == 'rnd':
            return _rnd_mixnorm(props, mus, covs, rng, n_samples=param)
        elif query == 'kurt':
            return _krt_mixnorm(props, mus, covs)
        else:
            raise ValueError('Invalid value of query: %s' % query)

    # --- mixture of 4 Gaussians, nonsymmetric and multimodal ---
    elif dist_name == 'p':
        props = np.array([1., 1., 2., 1.])
        props = props / np.sum(props)
        mus = np.array([-1, .3, -.3, 1.1])
        covs = np.array([.2, .2, .2, .2])

        if query == 'name':
            return 'Mix4Gauss_AssymMultiModal'
        elif query == 'pdf':
            return _pdf_mixnorm(props, mus, covs, xs=param_)
        elif query == 'rnd':
            return _rnd_mixnorm(props, mus, covs, rng, n_samples=param)
        elif query == 'kurt':
            return _krt_mixnorm(props, mus, covs)
        else:
            raise ValueError('Invalid value of query: %s' % query)

    # ---- mixture of 4 Gaussians, nonsymmetric and transitional ----
    elif dist_name == 'q':
        props = np.array([1., 3., 2., .5])
        props = props / np.sum(props)
        mus = np.array([-1, -.2, .3, 1])
        covs = np.array([.2, .3, .2, .2])

        if query == 'name':
            return 'Mix4Gauss_AssymTransitional'
        elif query == 'pdf':
            return _pdf_mixnorm(props, mus, covs, xs=param_)
        elif query == 'rnd':
            return _rnd_mixnorm(props, mus, covs, rng, n_samples=param)
        elif query == 'kurt':
            return _krt_mixnorm(props, mus, covs)
        else:
            raise ValueError('Invalid value of query: %s' % query)

    # ---- mixture of 4 Gaussians, nonsymmetric and unimodal ----
    elif dist_name == 'r':
        props = np.array([1., 2., 2., 1.])
        props = props / np.sum(props)
        mus = np.array([-.8, -.2, .2, .5])
        covs = np.array([.22, .3, .3, .2])

        if query == 'name':
            return 'Mix4Gauss_AssymUniModal'
        elif query == 'pdf':
            return _pdf_mixnorm(props, mus, covs, xs=param_)
        elif query == 'rnd':
            return _rnd_mixnorm(props, mus, covs, rng, n_samples=param)
        elif query == 'kurt':
            return _krt_mixnorm(props, mus, covs)
        else:
            raise ValueError('Invalid value of query: %s' % query)

    # ---- Student T with 3 degrees of freedom ----
    elif dist_name == 'a':
        if query == 'name':
            return 'Student_3deg'
        elif query == 'pdf':
            return stats.t.pdf(x=param_, df=3)
        elif query == 'rnd':
            return rng.standard_t(df=3, size=param)
        elif query == 'kurt':
            return np.inf
        else:
            raise ValueError('Invalid value of query: %s' % query)

    # ---- Student T with 5 degrees of freedom ----
    elif dist_name == 'd':
        if query == 'name':
            return 'Student_5deg'
        elif query == 'pdf':
            return stats.t.pdf(x=param_, df=5)
        elif query == 'rnd':
            return rng.standard_t(df=5, size=param)
        elif query == 'kurt':
            return 6.
        else:
            raise ValueError('Invalid value of query: %s' % query)

    # ---- Simple exponential ----
    elif dist_name == 'e':
        if query == 'name':
            return 'Exponential'
        elif query == 'pdf':
            tmp = np.array([1 if -1 < x else 0 for x in param_])
            return tmp * np.exp(-(param_ + 1))
        elif query == 'rnd':
            return -1 + rng.exponential(scale=1, size=param)
        elif query == 'kurt':
            return 6.
        else:
            raise ValueError('Invalid value of query: %s' % query)

    # ---- Double exponential ----
    elif dist_name == 'b':
        if query == 'name':
            return 'DbleExponential'
        elif query == 'pdf':
            return _pdf_dblexp(0, 1, xs=param_)
        elif query == 'rnd':
            return _rnd_dblexp(0, 1, rng, n_samples=param)
        elif query == 'kurt':
            return 3.
        else:
            raise ValueError('Invalid value of query: %s' % query)

    # ---- mixtures of 2 double exponential ----
    elif dist_name == 'f':
        props = np.array([.5, .5])
        mus = np.array([-1, 1])
        covs = np.array([ .5, .5])

        if query == 'name':
            return 'Mix2DbleExp'
        elif query == 'pdf':
            return _pdf_mixdblexp(props, mus, covs, xs=param_)
        elif query == 'rnd':
            return _rnd_mixdblexp(props, mus, covs, rng, n_samples=param)
        elif query == 'kurt':
            return _krt_mixdblexp(props, mus, covs)
        else:
            raise ValueError('Invalid value of query: %s' % query)

    # ---- Uniform ----
    elif dist_name == 'c':
        if query == 'name':
            return 'Uniform'
        elif query == 'pdf':
            tmp = np.array(
                [1 if np.abs(x) < np.sqrt(3) else 0 for x in param_])
            return 0.5 / np.sqrt(3) * tmp
        elif query == 'rnd':
            return 2 * np.sqrt(3) * rng.rand(param) - np.sqrt(3)
        elif query == 'kurt':
            return -1.2
        else:
            raise ValueError('Invalid value of query: %s' % query)

    # ---- mixture of 2 Gaussians, symmetric and multimodal ----
    elif dist_name == 'g':
        props = np.array([.5, .5])
        mus = np.array([-.5, .5])
        covs = np.array([.15, .15])

        if query == 'name':
            return 'Mix2Gauss_SymMultiModal'
        elif query == 'pdf':
            return _pdf_mixnorm(props, mus, covs, xs=param_)
        elif query == 'rnd':
            return _rnd_mixnorm(props, mus, covs, rng, n_samples=param)
        elif query == 'kurt':
            return _krt_mixnorm(props, mus, covs)
        else:
            raise ValueError('Invalid value of query: %s' % query)

    # ---- mixture of 2 Gaussians, symmetric and transitional ----
    elif dist_name == 'h':
        props = np.array([.5, .5])
        mus = np.array([-.5, .5])
        covs = np.array([.4, .4])

        if query == 'name':
            return 'Mix2Gauss_SymTransitional'
        elif query == 'pdf':
            return _pdf_mixnorm(props, mus, covs, xs=param_)
        elif query == 'rnd':
            return _rnd_mixnorm(props, mus, covs, rng, n_samples=param)
        elif query == 'kurt':
            return _krt_mixnorm(props, mus, covs)
        else:
            raise ValueError('Invalid value of query: %s' % query)

    # ---- mixture of 2 Gaussians, symmetric and unimodal ----
    elif dist_name == 'i':
        props = np.array([.5, .5])
        mus = np.array([-.5, .5])
        covs = np.array([.5, .5])

        if query == 'name':
            return 'Mix2Gauss_SymUniModal'
        elif query == 'pdf':
            return _pdf_mixnorm(props, mus, covs, xs=param_)
        elif query == 'rnd':
            return _rnd_mixnorm(props, mus, covs, rng, n_samples=param)
        elif query == 'kurt':
            return _krt_mixnorm(props, mus, covs)
        else:
            raise ValueError('Invalid value of query: %s' % query)

    # ---- mixture of 2 Gaussians, nonsymmetric and multimodal ----
    elif dist_name == 'j':
        props = np.array([1., 3.])
        props = props / np.sum(props)
        mus = np.array([-.5, .5])
        covs = np.array([.15, .15])

        if query == 'name':
            return 'Mix2Gauss_AssymMultiModal'
        elif query == 'pdf':
            return _pdf_mixnorm(props, mus, covs, xs=param_)
        elif query == 'rnd':
            return _rnd_mixnorm(props, mus, covs, rng, n_samples=param)
        elif query == 'kurt':
            return _krt_mixnorm(props, mus, covs)
        else:
            raise ValueError('Invalid value of query: %s' % query)

    # ---- mixture of 2 Gaussians, nonsymmetric and multimodal ----
    elif dist_name == 'k':
        props = np.array([1., 2.])
        props = props / np.sum(props)
        mus = np.array([-.7, .5])
        covs = np.array([.4, .4])

        if query == 'name':
            return 'Mix2Gauss_AssymTransitional'
        elif query == 'pdf':
            return _pdf_mixnorm(props, mus, covs, xs=param_)
        elif query == 'rnd':
            return _rnd_mixnorm(props, mus, covs, rng, n_samples=param)
        elif query == 'kurt':
            return _krt_mixnorm(props, mus, covs)
        else:
            raise ValueError('Invalid value of query: %s' % query)

    # ---- mixture of 2 Gaussians, nonsymmetric and unimodal ----
    elif dist_name == 'l':
        props = np.array([1., 2.])
        props = props / np.sum(props)
        mus = np.array([-.7, .5])
        covs = np.array([.5, .5])

        if query == 'name':
            return 'Mix2Gauss_AssymUniModal'
        elif query == 'pdf':
            return _pdf_mixnorm(props, mus, covs, xs=param_)
        elif query == 'rnd':
            return _rnd_mixnorm(props, mus, covs, rng, n_samples=param)
        elif query == 'kurt':
            return _krt_mixnorm(props, mus, covs)
        else:
            raise ValueError('Invalid value of query: %s' % query)

    # ---- Gaussian distribution ----
    elif dist_name == 's':
        props = np.array([1.])
        mus = np.array([0.])
        covs = np.array([.4])

        if query == 'name':
            return 'Gauss'
        elif query == 'pdf':
            return _pdf_mixnorm(props, mus, covs, xs=param_)
        elif query == 'rnd':
            return _rnd_mixnorm(props, mus, covs, rng, n_samples=param)
        elif query == 'kurt':
            return _krt_mixnorm(props, mus, covs)
        else:
            raise ValueError('Invalid value of query: %s' % query)

"""
Functions for double exponential distribution
"""

def _pdf_dblexp(mu, cov, xs):
    """Return pdf of double exponential distributions. 

    xs is assumed to be a scalar or vector, not matrix. 

    """
    return np.exp(-np.sqrt(2) * np.abs(xs - mu) / cov) / (np.sqrt(2) * cov)

def _rnd_dblexp(mu, cov, rng, n_samples):
    """Return random variables from double exponential distributions.
    """
    return (np.sign(rng.rand(n_samples) -.5) * 
        rng.exponential(scale=1 / np.sqrt(2), size=n_samples) * cov) + mu

def _krt_dblexp(mu, cov):
    """Return kurtosis of double exponential distributions
    """
    mu_ = mu * cov
    x1 = mu_
    x2 = mu_**2 + cov
    x3 = 3 * mu_ * cov**2 + mu_**3
    x4 = 6 * covs**2 + 6 * cov * mu_**2 + mu_**4
    return (x4 - 4 * x1 * x3 + 6 * x1**2 * x2 - 3 * x1**4) / \
            (x2 - x1**2)**2 - 3



"""
Functions for mixture of double exponential distributions
"""

def _pdf_mixdblexp(props, mus, covs, xs):
    """Return pdf of mixture of double exponential distributions. 

    xs is assumed to be a scalar or vector, not matrix. 

    """
    # ---- Get (n_components, n_samples) random matrix ----
    pdfs = np.vstack(
        [prop * _pdf_dblexp(mu, cov, xs)
         for prop, mu, cov in zip(props, mus, covs)])

    # ---- Return vector obtained by adding the random matrix ----
    return np.sum(pdfs, axis=0)

def _rnd_mixdblexp(props, mus, covs, rng, n_samples):
    """Return random variables from mixture of double exponential distributions.
    """
    # ---- Randomly select components ----
    # n_comps = len(mus)
    # comps = rng.randint(0, high=n_comps, size=n_samples)
    comps = rnd_discrete(props, rng, n_samples)

    # ---- Generate samples from selected components ----
    stds = np.sqrt(covs)
    return np.array(
        [_rnd_dblexp(mus[c], stds[c], rng, 1) for c in comps]).reshape(-1)

def _krt_mixdblexp(props, mus, covs):
    """Return kurtosis of mixture of normal distributions
    """
    mus_ = mus * covs
    x1 = np.sum(props * mus_)
    x2 = np.sum(props * (mus_**2 + covs))
    x3 = np.sum(props * (3 * mus_ * covs + mus_**3))
    x4 = np.sum(props * (6 * covs**2 + 6 * covs * mus_**2 + mus_**4))
    return (x4 - 4 * x1 * x3 + 6 * x1**2 * x2 - 3 * x1**4) / \
            (x2 - x1**2)**2 - 3



"""
Functions for mixture of normal distributions
"""

def _pdf_mixnorm(props, mus, covs, xs):
    """Return pdf of mixture of normal distributions. 

    xs is assumed to be a scalar or vector, not matrix. 

    """
    # ---- Get (n_components, n_samples) random matrix ----
    pdfs = np.vstack(
        [prop * stats.norm.pdf(xs, mu, cov)
         for prop, mu, cov in zip(props, mus, covs)])

    # ---- Return vector obtained by adding the random matrix ----
    return np.sum(pdfs, axis=0)

def _rnd_mixnorm(props, mus, covs, rng, n_samples):
    """Return random variables from mixture of normal distributions.
    """
    # ---- Randomly select components ----
    # n_comps = len(mus)
    # comps = rng.randint(0, high=n_comps, size=n_samples)
    comps = rnd_discrete(props, rng, n_samples)

    # ---- Generate samples from selected components ----
    return np.array(
        [rng.normal(mus[c], covs[c], 1) for c in comps]).reshape(-1)

def _krt_mixnorm(props, mus, covs):
    """Return kurtosis of mixture of normal distributions
    """
    x1 = np.sum(props * mus)
    x2 = np.sum(props * (mus**2 + covs**2))
    x3 = np.sum(props * (3 * mus * covs**2 + mus**3))
    x4 = np.sum(props * (3 * covs**4 + 6 * covs**2 * mus**2 + mus**4))
    return (x4 - 4 * x1 * x3 + 6 * x1**2 * x2 - 3 * x1**4) / \
            (x2 - x1**2)**2 - 3.



"""
Utility functions
"""

def rnd_discrete(props, rng, n_samples):
    """
    Generate samples from discrete distributions
    """
    cum = np.cumsum(props)
    return np.digitize(rng.rand(n_samples), cum).astype(int)

def wrap_array(x):
    """Wrap x as ndarray only if x is not ndarray. 

    This function may be moved to a utility module. 

    """
    if isinstance(x, collections.Iterable):
        if isinstance(x, np.ndarray):
            return x
        else:
            return np.array(x)
    else:
        return np.array([x])

