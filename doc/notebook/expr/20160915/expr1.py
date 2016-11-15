# -*- coding: utf-8 -*-

"""This module includes functions for evaluating accuracy of causal inference.

This module is used in 20160822-eval-bml.ipynb. 
"""

from copy import deepcopy
import numpy as np
import time

from bmlingam import do_mcmc_bmlingam, InferParams, MCMCParams, \
     load_pklz, define_hparam_searchspace, find_best_model
from bmlingam.utils.gendata import GenDataParams, gen_artificial_data


# A base parameter set for generating artificial data
gen_data_params_default = GenDataParams(
    n_samples=100, 
    b21_dist='r2intervals', 
    mu1_dist='randn', 
    mu2_dist='randn', 
    f1_scale=1.0, 
    f2_scale=1.0, 
    f1_coef=['r2intervals', 'r2intervals', 'r2intervals'], 
    f2_coef=['r2intervals', 'r2intervals', 'r2intervals'], 
    conf_dist=[['all'], ['all'], ['all']], 
    e1_std=3.0, 
    e2_std=3.0, 
    e1_dist=['laplace'], 
    e2_dist=['laplace'],
    seed=0
)


# A base parameter for causal inference
infer_params = InferParams(
    seed=0, 
    standardize=True, 
    subtract_mu_reg=False, 
    fix_mu_zero=True, 
    prior_var_mu='auto', 
    prior_scale='uniform', 
    max_c=1.0, 
    n_mc_samples=10000, 
    dist_noise='laplace', 
    df_indvdl=8.0, 
    prior_indvdls=['t'], 
    cs=[0.4, 0.6, 0.8],
    scale_coeff=2. / 3., 
    L_cov_21s=[-0.8, -0.6, -0.4, 0.4, 0.6, 0.8], 
    betas_indvdl=None, # [0.25, 0.5, 0.75, 1.], 
    betas_noise=[0.25, 0.5, 1.0, 3.0], 
    causalities=[[1, 2], [2, 1]], 
    sampling_mode='cache_mp4'
)


# A base parameters for MCMC
mcmc_params = MCMCParams(
    n_burn=1000, 
    n_mcmc_samples=1000
)


def gen_artificial_data_given_cond(ix_trial, cond):
    """Generate artificial data for given conditions (parameters).
    """
    # Set parameters for generating artificial data
    n_confs = cond['n_confs']
    gen_data_params = deepcopy(gen_data_params_default)
    gen_data_params.n_samples = cond['n_samples']
    gen_data_params.conf_dist = [['all'] for _ in range(n_confs)]
    gen_data_params.e1_dist = [cond['data_noise_type']]
    gen_data_params.e2_dist = [cond['data_noise_type']]

    noise_scale = cond['totalnoise'] / np.sqrt(n_confs)
    gen_data_params.f1_coef = [noise_scale for _ in range(n_confs)]
    gen_data_params.f2_coef = [noise_scale for _ in range(n_confs)]

    # Generate artificial data
    gen_data_params.seed = ix_trial
    data = gen_artificial_data(gen_data_params)
    
    return data


def estimate_hparams(xs, infer_params):
    """Estimate hyperparameters with the largest marginal likelihood value.
    """
    assert(type(infer_params) == InferParams)

    sampling_mode = infer_params.sampling_mode
    hparamss = define_hparam_searchspace(infer_params)
    results = find_best_model(xs, hparamss, sampling_mode)
    hparams_best = results[0]
    bf = results[2] - results[5] # Bayes factor
    
    return hparams_best, bf


def run_trial(ix_trial, cond):
    """Return results of causal inference using BMLiNGAM.

    Given conditions, i.e., parameters of generating artificial data and
    causal inference, a trial do causal inference and compute its accuracy. 
    """
    # Generate artificial data
    data = gen_artificial_data_given_cond(ix_trial, cond)
    b_true = data['b']
    causality_true = data['causality_true']
    
    # Causal inference
    t = time.time()
    infer_params.L_cov_21s = cond['L_cov_21s']
    infer_params.dist_noise = cond['model_noise_type']
    hparams, bf = estimate_hparams(data['xs'], infer_params)
    causality_est = hparams['causality']
    time_causal_inference = time.time() - t

    # 回帰係数推定
    t = time.time()
    trace = do_mcmc_bmlingam(data['xs'], hparams, mcmc_params)
    b_post = np.mean(trace['b'])
    time_posterior_inference = time.time() - t
    
    return {
        'causality_true': causality_true, 
        'regcoef_true': b_true, 
        'n_samples': cond['n_samples'], 
        'n_confs': cond['n_confs'], 
        'data_noise_type': cond['data_noise_type'], 
        'model_noise_type': cond['model_noise_type'], 
        'causality_est': causality_est,
        'correct_rate': (1.0 if causality_est == causality_true else 0.0), 
        'error_reg_coef': np.abs(b_post - b_true), 
        'regcoef_est': b_post, 
        'log_bf': 2 * bf, # 2log(p(M) / p(M_rev))なので常に正の値となります. 
        'time_causal_inference': time_causal_inference, 
        'time_posterior_inference': time_posterior_inference, 
        'L_cov_21s': str(cond['L_cov_21s']), 
        'n_mc_samples': infer_params.n_mc_samples, 
        'confs_absmean': np.mean(np.abs(data['confs'].ravel())), 
        'totalnoise': cond['totalnoise']
    }