# -*- coding: utf-8 -*-

"""Include functions to create cache of random sapmles for MC. 
"""
# Author: Taku Yoshioka
# License: MIT

from itertools import chain
import numpy as np
from parse import parse

from bmlingam.prob import sample_gg, multivariatet

flt = lambda l: list(chain.from_iterable(l))

def _is_uniform(L_cov_21):
    if type(L_cov_21) is not str:
        return False
    elif parse('U({:f},{:f})', L_cov_21.replace(' ', '')) is not None:
        return True
    else:
        raise ValueError("Invalid value of L_cov_21: %s" % L_cov_21)

def _get_L_cov(L_cov_21, rng):
    if type(L_cov_21) in (float, int):
        return np.array([[1.0, L_cov_21], 
                         [L_cov_21, 1.0]])
    elif _is_uniform(L_cov_21):
        r = parse('U({:f},{:f})', L_cov_21.replace(' ', ''))
        L_cov_21 = rng.uniform(r[0], r[1])
        return np.array([[1.0, L_cov_21], 
                         [L_cov_21, 1.0]])

def _cache_mu_indvdl(
    prior_indvdls, L_covs, df_indvdls, beta_coeffs, n_mc_samples, n_samples, 
    rng):
    # Define conditional parameter sets
    df_indvdls_ = lambda p: df_indvdls if p == 't' else [None]
    beta_coeffs_ = lambda p: beta_coeffs if p == 'gg' else [None]

    # Make list of parameter sets
    paramss = flt([
        [
            (prior_indvdl, L_cov, df_indvdl, beta)
            for df_indvdl in df_indvdls_(prior_indvdl)
            for beta in beta_coeffs_(prior_indvdl)
        ]
        for prior_indvdl in prior_indvdls
        for L_cov in L_covs
    ])

    # Define function to draw random samples
    def f(params):
        prior_indvdl = params[0]
        L_covs = [_get_L_cov(params[1], rng) for _ in range(n_mc_samples)]
        df_L = params[2]
        beta = params[3]

        if prior_indvdl == 't':
            sample_func = lambda L_cov_: multivariatet(
                0, L_cov_, df_L, n_samples, rng) / np.sqrt(df_L / (df_L - 2))
        elif prior_indvdl == 'gauss':
            sample_func = lambda L_cov_: rng.multivariate_normal(
                np.zeros(L_cov_.shape[0]), L_cov_, n_samples)
        elif prior_indvdl == 'gg':
            sample_func = lambda L_cov_: sample_gg(
                L_cov_, beta, n_samples, rng)
        else:
            raise ValueError(
                'Invalid value of prior_indvdl: {}'.format(prior_indvdl))

        return np.vstack([sample_func(L_cov) for L_cov in L_covs])

    # Draw random samples
    mus = [f(params) for params in paramss]

    # Split into components
    mu_r = lambda r, mu: mu[:, r - 1].reshape(n_mc_samples, n_samples)
    key_r = lambda r, params: (
        'mu_indvdl{}'.format(r), params[0], params[1], params[2], params[3])

    mu_indvdl1 = {key_r(1, params): mu_r(1, mu)
                  for (params, mu) in zip(paramss, mus)}
    mu_indvdl2 = {key_r(2, params): mu_r(2, mu)
                  for (params, mu) in zip(paramss, mus)}

    # Comcbine dicts
    mu_indvdl1.update(mu_indvdl2)

    return mu_indvdl1

def _cache_mu1(n_mc_samples, n_samples, rng):
    zeros = np.zeros((n_mc_samples, n_samples))
    randnones = rng.randn(n_mc_samples, 1) * np.ones((1, n_samples))

    return {
        ('mu1', True): zeros, 
        ('mu1', False): randnones
    }

def _cache_mu2(n_mc_samples, n_samples, rng):
    zeros = np.zeros((n_mc_samples, n_samples))
    randnones = rng.randn(n_mc_samples, 1) * np.ones((1, n_samples))

    return {
        ('mu2', True): zeros, 
        ('mu2', False): randnones
    }

def _cache_h1(n_mc_samples, n_samples, rng):
    randnones = rng.randn(n_mc_samples, 1) * np.ones((1, n_samples))
    logn_ones = rng.lognormal(size=(n_mc_samples, 1)) * np.ones((1, n_samples))
    uniformn_ones = rng.uniform(size=(n_mc_samples, 1)) * np.ones((1, n_samples))

    return {
        ('h1', 'tr_normal'): np.abs(randnones), 
        ('h1', 'log_normal'): logn_ones, 
        ('h1', 'uniform'): uniformn_ones
    }

def _cache_h2(n_mc_samples, n_samples, rng):
    randnones = rng.randn(n_mc_samples, 1) * np.ones((1, n_samples))
    logn_ones = rng.lognormal(size=(n_mc_samples, 1)) * np.ones((1, n_samples))
    uniformn_ones = rng.uniform(size=(n_mc_samples, 1)) * np.ones((1, n_samples))

    return {
        ('h2', 'tr_normal'): np.abs(randnones), 
        ('h2', 'log_normal'): logn_ones, 
        ('h2', 'uniform'): uniformn_ones
    }

def _cache_b(n_mc_samples, n_samples, rng):
    randnones = rng.randn(n_mc_samples, 1) * np.ones((1, n_samples))
    return {'b': randnones}

def create_cache_source(xs, hparamss):
    """Create the source of shared cache. 
    """
    n_mc_samples = hparamss[0]['n_mc_samples']
    rng = np.random.RandomState(hparamss[0]['seed'])
    n_samples = len(xs)

    prior_indvdls = set([h['prior_indvdl'] for h in hparamss])
    L_cov_21s = set([h['L_cov_21'] for h in hparamss])
    df_indvdls = set([h['df_indvdl'] for h in hparamss])
    beta_coeffs = set([h['beta_coeff'] for h in hparamss])
    # fix_mu_zeros = set([h['fix_mu_zero'] for h in hparamss])
    # prior_scales = set([h['prior_scale'] for h in hparamss])

    cache_source = {}
    cache_source.update(_cache_mu_indvdl(
        prior_indvdls, L_cov_21s, df_indvdls, beta_coeffs, n_mc_samples, 
        n_samples, rng))
    cache_source.update(_cache_mu1(n_mc_samples, n_samples, rng))
    cache_source.update(_cache_mu2(n_mc_samples, n_samples, rng))
    cache_source.update(_cache_h1(n_mc_samples, n_samples, rng))
    cache_source.update(_cache_h2(n_mc_samples, n_samples, rng))
    cache_source.update(_cache_b(n_mc_samples, n_samples, rng))

    return cache_source

def fetch_mu_indvdl(
    cache, prior_indvdl, L_cov_21, df_indvdl, beta_coeff):
    mu_indvdl1_ = cache[
        ('mu_indvdl1', prior_indvdl, L_cov_21, df_indvdl, beta_coeff)]
    mu_indvdl2_ = cache[
        ('mu_indvdl2', prior_indvdl, L_cov_21, df_indvdl, beta_coeff)]

    return mu_indvdl1_, mu_indvdl2_

def fetch_mu1(cache, fix_mu_zero):
    return cache[('mu1', fix_mu_zero)]

def fetch_mu2(cache, fix_mu_zero):
    return cache[('mu2', fix_mu_zero)]

def fetch_h1(cache, prior_scale):
    return cache[('h1', prior_scale)]

def fetch_h2(cache, prior_scale):
    return cache[('h2', prior_scale)]

def fetch_b(cache):
    return cache['b']

