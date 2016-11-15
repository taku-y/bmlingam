# -*- coding: utf-8 -*-

"""Chache for MC sampling using shared memory.
"""
# Author: Taku Yoshioka
# License: MIT

import ctypes
import multiprocessing
import numpy as np
from scipy.misc import logsumexp

import six

from bmlingam.bmlingam_np import _comp_exp_nakami_xis_givenMgammatheta
from bmlingam.cache_mc import create_cache_source, fetch_mu_indvdl, \
                              fetch_mu1, fetch_mu2, fetch_h1, fetch_h2, \
                              fetch_b
from bmlingam.utils import standardize_samples

xs_local = None
cache_local = None

def _comp_logP(proc_params):
    """Compute :math:`\\log p({\\cal M}_{r}|{\\cal D}) (r=1 or 2)` 
    for given model (up to constant :math:`\\log p({\\cal D})`). 

    Integration over :math:`\\theta` is performed by naive MonteCalro 
    sampling. :code:`hparams` should have field `causality` ([1, 2] or [2, 1]).
    """
    xs = xs_local
    hparams = proc_params['hparams']

    causality    = hparams['causality']
    prior_indvdl = hparams['prior_indvdl']
    L_cov_21     = hparams['L_cov_21']
    df_indvdl    = hparams['df_indvdl']
    beta_coeff   = hparams['beta_coeff']
    fix_mu_zero  = hparams['fix_mu_zero']
    prior_scale  = hparams['prior_scale']
    n_mc_samples = hparams['n_mc_samples']
    standardize  = hparams['standardize']
    scale_coeff  = hparams['scale_coeff']

    # ---- Standardization ----
    xs = standardize_samples(xs, standardize)

    # ---- Scaling parameters ----
    std_x = np.std(xs, axis=0)
    max_c = hparams['max_c']
    tau_cmmn = np.array([(std_x[0] * max_c)**2, (std_x[1] * max_c)**2])
    del max_c

    # ---- Stochastic variables ----
    # Individual specific effects
    mu_indvdl1_, mu_indvdl2_ = fetch_mu_indvdl(
        cache_local, prior_indvdl, L_cov_21, df_indvdl, beta_coeff)
    mu_indvdl1 = std_x[0] * hparams['v_indvdl_1'] * mu_indvdl1_
    mu_indvdl2 = std_x[1] * hparams['v_indvdl_2'] * mu_indvdl2_

    # Mean
    mu1_ = fetch_mu1(cache_local, fix_mu_zero)
    mu2_ = fetch_mu2(cache_local, fix_mu_zero)
    if hparams['prior_var_mu'] == 'auto':
        mu1 = np.sqrt(tau_cmmn[0]) * mu1_
        mu2 = np.sqrt(tau_cmmn[1]) * mu2_
    else:
        v = hparams['prior_var_mu']
        mu1 = np.sqrt(v) * mu1_
        mu2 = np.sqrt(v) * mu2_

    # Noise variance
    h1_ = fetch_h1(cache_local, prior_scale)
    h2_ = fetch_h2(cache_local, prior_scale)
    h1 = np.sqrt(tau_cmmn[0]) * h1_
    h2 = np.sqrt(tau_cmmn[1]) * h2_

    # Regression coefficient
    b_ = scale_coeff * fetch_b(cache_local)
    if causality == [1, 2]: # x1 -> x2
        b = np.sqrt(tau_cmmn[1]) * b_
    else: # x2 -> x1
        b = np.sqrt(tau_cmmn[0]) * b_

    # ---- Compute log probability ----
    if causality == [1, 2]:
        e = _comp_exp_nakami_xis_givenMgammatheta(
            1, n_mc_samples, xs, (mu1 + mu_indvdl1), (mu2 + mu_indvdl2), 
            b, None, h1, h2, hparams)
        prior_model = np.log(hparams['P_M1'])
    else:
        e = _comp_exp_nakami_xis_givenMgammatheta(
            2, n_mc_samples, xs, (mu1 + mu_indvdl1), (mu2 + mu_indvdl2), 
            None, b, h1, h2, hparams)
        prior_model = np.log(hparams['P_M2'])

    logps = np.sum(e, axis=1) # Sum logp over samples
    logp = logsumexp(logps, 0)  + prior_model - np.log(n_mc_samples)

    return logp

    # traces = {
    #     'mu1': mu1[:, 0], 
    #     'mu2': mu2[:, 0], 
    #     'h1': h1[:, 0], 
    #     'h2': h2[:, 0], 
    #     'mu1_': mu1_,
    #     'mu2_': mu2_, 
    #     'b': b, 
    #     'logps_M1': np.log(hparams['P_M1']), 
    #     'logps_M2': np.log(hparams['P_M2'])
    # }

    # return logp, traces

def _init_shared_local(cache_shared, xs_shared):
    """Initialize cache dict for MC sampling. 

    The cache dict initialized in this function is used on remote processes.
    This function is invoked where the worker pool is initialized in the 
    main process. 

    Input argument :code:`cache_shared` is a dict, whose entries have the 
    following format:

    .. code-block:: python

        {('h2', 'tr_normal'): (shared_array_base, shared_array_size)}

    The first is a tuple consisting of the variable name ('h2' in the above 
    example) and hyperparameter value ('tr_noemal'). The second is the 
    (reference to the) shared array, which includes random samples used in MC. 
    The last one is the size of the shared array. The size is needed because 
    the shared array does not have the information of the size. 

    For each entry, this function make an ndarray view for the shared array 
    and store it with the associated key. 
    """
    global cache_local

    cache_local = {}

    # for k, v in cache_shared.iteritems():
    for k, v in six.iteritems(cache_shared):
        shared_array_base, shape = v
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(shape)

        cache_local.update({k: shared_array})

    global xs_local

    xs_shared_base, shape = xs_shared
    xs_local = np.ctypeslib.as_array(xs_shared_base.get_obj())
    xs_local = xs_local.reshape(shape)

def _create_cache_shared(xs, hparamss):
    """Create shared cache. 
    """
    cache_source = create_cache_source(xs, hparamss)
    cache_shared = {}

    # for k, v in cache_source.iteritems():
    for k, v in six.iteritems(cache_source):
        assert(v.dtype == np.float32 or v.dtype == np.float64 or 
               v.dtype == float)
        n = len(v.reshape(-1))
        shared_array_base = multiprocessing.Array(ctypes.c_double, n)
        shape = v.shape

        view = np.ctypeslib.as_array(shared_array_base.get_obj())
        view = view.reshape(shape)
        view[:] = v[:]
        del view

        cache_shared.update({k: (shared_array_base, shape)})

    return cache_shared

def _create_xs_shared(xs):
    """Create shared variable for data (xs).
    """
    n = len(xs.reshape(-1))
    xs_shared_base = multiprocessing.Array(ctypes.c_double, n)
    shape = xs.shape

    view = np.ctypeslib.as_array(xs_shared_base.get_obj())
    view = view.reshape(shape)
    view[:] = xs[:]
    del view

    xs_shared = (xs_shared_base, shape)

    return xs_shared

def comp_logPs_mp(xs, hparamss, processes):
    cache_shared = _create_cache_shared(xs, hparamss)
    xs_shared = _create_xs_shared(xs)

    pool = multiprocessing.Pool(
        processes=processes, initializer=_init_shared_local, 
        initargs=(cache_shared, xs_shared))

    proc_paramss = [{
        'hparams': hparams
    } for hparams in hparamss]

    logPs = np.array(pool.map(_comp_logP, proc_paramss))
    pool.close()

    return logPs

