# -*- coding: utf-8 -*-

"""Hyperparameter optimization by grid search.
"""
# Author: Taku Yoshioka, Shohei Shimizu
# License: MIT

import numpy as np

from bmlingam.cache_mc import create_cache_source
from bmlingam.bmlingam_np import comp_logP, comp_logP_bmlingam_np
from bmlingam.cache_mc_shared import comp_logPs_mp

def find_best_model(xs, hparamss, sampling_mode='normal'):
    """Find the optimal and reverse-optimal models.

    This function calculates marginal likelihoods for the models specified with
    hyperparameter sets included in :code:`hparamss`. The optimal model is
    the one whose marginal likelihood on data :code:`xs` is the maximum.
    Reverse-optimal is the one with the maximum marginal likelihood across
    models whose causations are opposite to the optimal model.

    :param xs: Pairs of observed values.
    :type xs: numpy.ndarray, shape=(n_samples, 2)
    :param hparamss: List of model parameters which defines search space.
        This is created with :func:`bmlingam.define_hparam_searchspace`.
    :type hparamss: list of dict
    :param sampling_mode: Specify the way to perform Monte Carlo sampling. 
    :type sampling_mode: str
    """
    assert((xs.shape[1] == 2) and (xs.ndim == 2))
    assert(type(sampling_mode) is str)
    assert(type(hparamss) is list)

    if sampling_mode == 'normal':
        logPs = np.array([comp_logP_bmlingam_np(xs, hparams)[0]
                          for hparams in hparamss])
    elif sampling_mode == 'cache':
        cache = create_cache_source(xs, hparamss)
        logPs = np.array([comp_logP(xs, hparams, cache)[0]
                          for hparams in hparamss])
    elif sampling_mode == 'cache_mp2':
        logPs = comp_logPs_mp(xs, hparamss, processes=4)
    elif sampling_mode == 'cache_mp4':
        logPs = comp_logPs_mp(xs, hparamss, processes=4)
    elif sampling_mode == 'cache_mp8':
        logPs = comp_logPs_mp(xs, hparamss, processes=8)
    else:
        raise ValueError("Invalid value of sampling_mode: %s" % sampling_mode)

    ix_max_logP = np.argmax(logPs)

    # Find reverse-optimal model
    causality_est = hparamss[ix_max_logP]['causality']
    ixs_rev = np.array(
        [ix_model for ix_model in range(len(hparamss))
         if hparamss[ix_model]['causality'] != causality_est]).astype(int)
    assert(len(ixs_rev) == len(hparamss) / 2)
    ix_max_rev = ixs_rev[np.argmax(logPs[ixs_rev])]

    # Posterior probabilities
    exps = np.exp(logPs - np.max(logPs))
    posterior = 1. / np.sum(exps) # NOTE: exp(0) = 1
    posterior_rev = exps[ix_max_rev] / np.sum(exps)

    # Log-likelihoods
    ll = np.max(logPs)
    ll_rev = np.max(logPs[ix_max_rev])

    return hparamss[ix_max_logP], posterior,     ll, \
           hparamss[ix_max_rev],  posterior_rev, ll_rev
