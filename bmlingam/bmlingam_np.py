# -*- coding: utf-8 -*-

"""Implementation bayesian mixed LiNGAM model using numpy array.
"""
# Author: Taku Yoshioka, Shohei Shimizu
# License: MIT

from itertools import chain
import numpy as np
from scipy.misc import logsumexp

from bmlingam.hparam import define_hparam_searchspace
from bmlingam.prob import ll_laplace, ll_gg, sample_gg, multivariatet
from bmlingam.utils import standardize_samples
from bmlingam.cache_mc import fetch_mu_indvdl, fetch_mu1, fetch_mu2, \
                              fetch_h1, fetch_h2, fetch_b

flt = lambda l: list(chain.from_iterable(l))

def a2t(L_cov):
    return (L_cov[0, 0], L_cov[0, 1], L_cov[1, 1])

def t2a(L_cov):
    return np.array([[L_cov[0], L_cov[1]], 
                     [L_cov[1], L_cov[2]]])

def comp_logP(xs, hparams, cache):
    """Compute :math:`\\log p({\\cal M}_{r}|{\\cal D}) (r=1 or 2)` 
    for given model (up to constant :math:`\\log p({\\cal D})`). 

    Integration over :math:`\\theta` is performed by naive MonteCalro 
    sampling. :code:`hparams` should have field `causality` ([1, 2] or [2, 1]).
    """
    causality    = hparams['causality']
    prior_indvdl = hparams['prior_indvdl']
    L_cov_21     = hparams['L_cov_21']
    df_indvdl    = hparams['df_indvdl']
    beta_coeff   = hparams['beta_coeff']
    fix_mu_zero  = hparams['fix_mu_zero']
    prior_scale  = hparams['prior_scale']
    n_mc_samples = hparams['n_mc_samples']
    standardize  = hparams['standardize']

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
        cache, prior_indvdl, L_cov_21, df_indvdl, beta_coeff)
    mu_indvdl1 = std_x[0] * hparams['v_indvdl_1'] * mu_indvdl1_
    mu_indvdl2 = std_x[1] * hparams['v_indvdl_2'] * mu_indvdl2_

    # Mean
    mu1_ = fetch_mu1(cache, fix_mu_zero)
    mu2_ = fetch_mu2(cache, fix_mu_zero)
    if hparams['prior_var_mu'] == 'auto':
        mu1 = np.sqrt(tau_cmmn[0]) * mu1_
        mu2 = np.sqrt(tau_cmmn[1]) * mu2_
    else:
        v = hparams['prior_var_mu']
        mu1 = np.sqrt(v) * mu1_
        mu2 = np.sqrt(v) * mu2_

    # Noise variance
    h1_ = fetch_h1(cache, prior_scale)
    h2_ = fetch_h2(cache, prior_scale)
    h1 = np.sqrt(tau_cmmn[0]) * h1_
    h2 = np.sqrt(tau_cmmn[1]) * h2_

    # Regression coefficient
    b_ = fetch_b(cache)
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

    traces = {
        'mu1': mu1[:, 0], 
        'mu2': mu2[:, 0], 
        'h1': h1[:, 0], 
        'h2': h2[:, 0], 
        'mu1_': mu1_,
        'mu2_': mu2_, 
        'b': b, 
        'logps_M1': np.log(hparams['P_M1']), 
        'logps_M2': np.log(hparams['P_M2'])
    }

    return logp, traces

def comp_logP_bmlingam_np(xs, hparams):
    """Compute :math:`\\log p({\\cal M}_{r}|{\\cal D}) (r=1 or 2)` 
    for given model (up to constant :math:`\\log p({\\cal D})`). 

    Integration over :math:`\\theta` is performed by naive MonteCalro 
    sampling. :code:`hparams` should have field `causality` ([1, 2] or [2, 1]). 

    Return values logP_givenD is :math:`\\log p({\\cal D}|{\\cal M}_{r})`. 

    :param xs: Observed samples. 
    :type xs: ndarray, shape=(n_samples, 2), dtype=float
    :param hparams: Hyperparameters to be tested.
    :param np.random.RandomState rng: Random number generator. 
    :return: (logP_givenD, traces)
    :rtype: tuple
    """
    rng = np.random.RandomState(hparams['seed'])
    logP_M1_givenD, logP_M2_givenD, traces = comp_logP_M12_np(xs, hparams, rng)
    
    if hparams['causality'] == [1, 2]:
        traces.update({'b': traces['b21']})
        traces.update({'logps': traces['logps_M1']})
        return logP_M1_givenD, traces
    else:
        traces.update({'b': traces['b12']})
        traces.update({'logps': traces['logps_M2']})
        return logP_M2_givenD, traces

def comp_logP_M12_np(xs, hparams, rng):
    """Compute :math:`\\log p({\\cal M}_{r}|{\\cal D}) (r=1, 2)` 
    for given two models (up to constant :math:`\\log p({\\cal D})`). 

    Integration over :math:`\\theta` is performed by naive MonteCalro 
    sampling. 

    Return values logP_M1_givenD and logP_M2_givenD denotes
    :math:`\\log p({\\cal D}|{\\cal M}_{r})`

    :param xs: Observed samples. 
    :type xs: ndarray, shape=(n_samples, 2), dtype=float
    :param hparams: Hyperparameters to be tested.
    :param np.random.RandomState rng: Random number generator. 
    :return: logP_M1_givenD, logP_M2_givenD
    :rtype: (float, float)

    """
    n_mc_samples = hparams['n_mc_samples']
    standardize = hparams['standardize']

    # ---- Normalization ----
    xs = standardize_samples(xs, standardize)
    n_samples = xs.shape[0]

    # ---- Scaling parameters ----
    std_x = np.std(xs, axis=0)
    max_c = hparams['max_c']
    tau_cmmn = np.array([(std_x[0] * max_c)**2, (std_x[1] * max_c)**2])

    # ---- Individual-specific effects ----    
    # Do sampling from Gauss or T
    L_cov = hparams['L_cov']
    assert(np.min(np.diag(L_cov)) == 1.)
    assert(np.max(np.diag(L_cov)) == 1.)

    if hparams['prior_indvdl'] == 't':
        df_L = hparams['df_indvdl']
        L = multivariatet(0, L_cov, df_L, n_mc_samples * n_samples, rng)
        L /= np.sqrt(df_L / (df_L - 2)) # Normalize variance to one
    elif hparams['prior_indvdl'] == 'gauss':
        L = rng.multivariate_normal(
            np.zeros(L_cov.shape[0]), L_cov, n_mc_samples * n_samples)
    elif hparams['prior_indvdl'] == 'gg':
        beta = hparams['beta_coeff']
        L = sample_gg(L_cov, beta, n_mc_samples * n_samples, rng)
    else:
        raise ValueError('Invalid value of prior_indvdl: %s' %
            hparams['prior_indvdl'])
    del L_cov

    L_mu1 = L[:, 0].reshape(n_mc_samples, n_samples)
    L_mu2 = L[:, 1].reshape(n_mc_samples, n_samples)

    # Scale variance according to given hyperparameters and data std
    mu1_ = std_x[0] * hparams['v_indvdl_1'] * L_mu1
    mu2_ = std_x[1] * hparams['v_indvdl_2'] * L_mu2

    # ---- Sampling functions ----
    gen_randnones = lambda: (
        rng.randn(n_mc_samples, 1) * np.ones((1, n_samples)))
    gen_logn_ones = lambda: (
        rng.lognormal(size=(n_mc_samples, 1)) * np.ones((1, n_samples)))
    gen_uniform_ones = lambda: (
        rng.uniform(size=(n_mc_samples, 1)) * np.ones((1, n_samples)))

    # ---- Mean ----
    if hparams['fix_mu_zero']:
        mu1 = np.zeros((n_mc_samples, n_samples))
        mu2 = np.zeros((n_mc_samples, n_samples))
    elif hparams['prior_var_mu'] == 'auto':
        mu1 = np.sqrt(tau_cmmn[0]) * gen_randnones()
        mu2 = np.sqrt(tau_cmmn[1]) * gen_randnones()
    else:
        v = hparams['prior_var_mu']
        mu1 = np.sqrt(v) * gen_randnones()
        mu2 = np.sqrt(v) * gen_randnones()
        
    # ---- Noise variance ----
    if hparams['prior_scale'] == 'tr_normal':
        h1 = np.sqrt(tau_cmmn[0]) * np.abs(gen_randnones())
        h2 = np.sqrt(tau_cmmn[1]) * np.abs(gen_randnones())
    elif hparams['prior_scale'] == 'log_normal':
        h1 = np.sqrt(tau_cmmn[0]) * gen_logn_ones()
        h2 = np.sqrt(tau_cmmn[1]) * gen_logn_ones()
    elif hparams['prior_scale'] == 'uniform':
        h1 = np.sqrt(tau_cmmn[0]) * gen_uniform_ones()
        h2 = np.sqrt(tau_cmmn[1]) * gen_uniform_ones()
    else:
        raise ValueError(
            "Invalid value of prior_scale: %s" % hparams['prior_scale'])

    # ---- Regression coefficient ----
    b21 = np.sqrt(tau_cmmn[1]) * gen_randnones()
    b12 = np.sqrt(tau_cmmn[0]) * gen_randnones()
    del max_c, tau_cmmn

    # ---- Compute log probabilities of samples given parameters ----
    exp_nakami_xis_givenMgammatheta1 = \
        _comp_exp_nakami_xis_givenMgammatheta(
            1, n_mc_samples, xs, (mu1 + mu1_), (mu2 + mu2_), b21, None, h1, h2, 
            hparams)
    exp_nakami_xis_givenMgammatheta2 = \
        _comp_exp_nakami_xis_givenMgammatheta(
            2, n_mc_samples, xs, (mu1 + mu1_), (mu2 + mu2_), None, b12, h1, h2, 
            hparams)

    # ---- Averaging over MonteCarlo samples ----
    logps_M1 = (logsumexp(exp_nakami_xis_givenMgammatheta1, 0) -
                np.log(n_mc_samples))
    logps_M2 = (logsumexp(exp_nakami_xis_givenMgammatheta2, 0) - 
                np.log(n_mc_samples))
    logP_D_givenM1 = np.sum(logps_M1)    
    logP_D_givenM2 = np.sum(logps_M2)

    # ------ Log marginal probability (up to const) ------
    logP_M1_givenD = logP_D_givenM1 + np.log(hparams['P_M1'])
    logP_M2_givenD = logP_D_givenM2 + np.log(hparams['P_M2'])

    # ---- Sampling traces ----
    traces = {
        'mu1': mu1[:, 0], 
        'mu2': mu2[:, 0], 
        'h1': h1[:, 0], 
        'h2': h2[:, 0], 
        'mu1_': mu1_,
        'mu2_': mu2_, 
        'b12': b12[:, 0], 
        'b21': b21[:, 0], 
        'logps_M1': logps_M1, 
        'logps_M2': logps_M2
    }

    return logP_M1_givenD, logP_M2_givenD, traces

"""Supplemental functions
"""

def _comp_exp_nakami_xis_givenMgammatheta(
    M_id, n_theta_sampling, xs, mu1, mu2, b21, b12, h1, h2, hparams):
    """Compute probabilities of samples given parameters: 
    :math:`p(x_{l} - \\mu_{l} - b_{lm}x_{m}|{\\bf\\theta})`. 

    It is assumed a Laplace distribution as noise model. 

    :param int M_id: Model ID (1 or 2).
    :param int n_theta_sampling: Number of sampling iterations.
    :param xs: Observed samples. 
    :type xs: ndarray, shape=(n_samples, 2), dtype=float
    :param float mu1: Mean of the 1st signal.
    :param float mu2: Mean of the 2nd signal.
    :param float b21: Connection strength: 1 -> 2.
    :param float b12: Connection strength: 2 -> 1.
    :param float h1: Std of the 1st signal.
    :param float h2: Std of the 2nd signal.
    :return: Probabilities of samples given parameters. 
    :rtype: ndarray, shape=(n_theta_sampling, n_samples), dtype=float
    """
    gen_ones = lambda: np.ones((n_theta_sampling, 1))

    if hparams['dist_noise'] == 'laplace':
        logp = ll_laplace
    elif hparams['dist_noise'] == 'gg':
        logp = lambda e: ll_gg(e, hparams['beta_noise'])
    else:
        raise ValueError(
            "Invalid value of hparams['beta_noise']: %s" % 
            hparams['beta_noise'])

    if hparams['subtract_mu_reg']:
        if M_id == 1:
            exp_nakami_xis_givenMgammatheta = (
                logp((gen_ones() * xs[:, 0].T - mu1) / h1) +
                logp((gen_ones() * xs[:, 1].T - mu2 - 
                      b21 * (gen_ones() * xs[:, 0].T - mu1)) / h2))
        elif M_id == 2:
            exp_nakami_xis_givenMgammatheta = (
                logp((gen_ones() * xs[:, 0].T - mu1 -
                      b12 * (gen_ones() * xs[:, 1].T - mu2)) / h1) +
                logp((gen_ones() * xs[:, 1].T - mu2) / h2))
        else:
            raise ValueError('Invalid value of M_id: %s.' + str(M_id))
    else:
        if M_id == 1:
            exp_nakami_xis_givenMgammatheta = (
                logp((gen_ones() * xs[:, 0].T - mu1) / h1) +
                logp((gen_ones() * xs[:, 1].T - mu2 - 
                      b21 * (gen_ones() * xs[:, 0].T)) / h2))
        elif M_id == 2:
            exp_nakami_xis_givenMgammatheta = (
                logp((gen_ones() * xs[:, 0].T - mu1 -
                      b12 * (gen_ones() * xs[:, 1].T)) / h1) +
                logp((gen_ones() * xs[:, 1].T - mu2) / h2))
        else:
            raise ValueError('Invalid value of M_id: %s.' + str(M_id))

    exp_nakami_xis_givenMgammatheta = \
        exp_nakami_xis_givenMgammatheta - np.log(h1) - np.log(h2)

    return exp_nakami_xis_givenMgammatheta
