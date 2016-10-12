# -*- coding: utf-8 -*-

"""Evaluate accuracy of Bayesian mixed LiNGAM model.
"""
# Author: Taku Yoshioka, Shohei Shimizu
# License: MIT

import pandas as pd
import time

from bmlingam import define_hparam_searchspace, find_best_model, InferParams
from bmlingam.tests.gendata import gen_samples, GenDataParams

"""Accuracy of model selection (causality inference)
"""

def eval_find_best_model(
    n_confounderss=[0, 1, 6, 12], n_trials=10, n_samples=100, 
    min_correctss=[6, 6, 6, 6], prior_indvdlss=[['t'], ['gauss'], ['gg']], 
    dists_noise=['laplace', 'gg'], show_progress=False, show_results=True, 
    betas_indvdl=[.25, .5, .75, 1.], betas_noise=[.25, .5, .75, 1.], 
    standardize=False, sample_coef='r2intervals', n_mc_samples=10000, 
    sampling_mode='normal'):
    """Test estimation using Bayesian mixed LiNGAM model.

    The tests run over numbers of confounders: 0, 1, 6 and 12. 
    Each of the tests passes if the ratio of correct estimations is 
    greater than a threshold for each of settings. 

    The testing parameters are as follows:

    .. code:: python

        n_confounderss = [0, 1, 6, 12] # Number of confounders
        n_trials = 10 # Number of trials (inferences)
        min_corrects = [6, 6, 6, 6] # Lower threshold of correct inferences
        n_samples = 100 # Number of observations

    The default set of hyperparameters are used to do empirical Bayesian 
    estimation (:py:func:`lingam.define_hparam_searchspace`). 

    Input argument :code:`tied_sampling` is for backward compatibility to 
    the original implementation. 
    """
    # ---- Program started ----
    t_start = time.time()
    print('Program started at %s\n' % time.strftime("%c"))

    print('Test parameters')
    print('  n_confounderss: %s' % str(n_confounderss))
    print('  n_samples     : %d' % n_samples)
    print('  min_correctss : %s' % str(min_correctss))
    print('  sampling_mode : %s' % str(sampling_mode))
    print('')
    print('Model search space')
    print('  prior_indvdlss: %s' % prior_indvdlss)
    print('  dists_noise   : %s' % dists_noise)
    print('  betas_indvdl  : %s' % str(betas_indvdl))
    print('  betas_noise   : %s' % str(betas_noise))
    print('  standardize   : %s' % str(standardize))
    print('')

    # ---- Test parameters ----
    test_paramss = [
        {
            'n_trials': n_trials,
            'min_corrects': min_corrects, 
            'gen_data_params': GenDataParams(
                n_confounders=n_confounders, 
                n_samples=n_samples, 
                sample_coef=sample_coef
            )
        }
        for (n_confounders, min_corrects) in zip(n_confounderss, min_correctss)
    ]

    # ---- Hyperparameter search spaces ----
    hparamsss = [
        define_hparam_searchspace(
            InferParams(
                standardize=standardize,
                n_mc_samples=n_mc_samples, 
                prior_indvdls=prior_indvdls, 
                dist_noise=dist_noise, 
                betas_indvdl=betas_indvdl, 
                betas_noise=betas_noise 
            )
        )
        for prior_indvdls in prior_indvdlss
        for dist_noise in dists_noise
    ]

    # ---- Loop over experimental conditions ----
    n_confounders_ = []
    prior_indvdl_ = []
    dist_noise_ = []
    n_corrects_ = []

    for i, test_params in enumerate(test_paramss):
        for j, hparamss in enumerate(hparamsss):
            if show_progress:
                t_start_local = time.time()
                print('---- test_params (%d/%d), hparamss (%d/%d) ----' %
                    (i + 1, len(test_paramss), j + 1, len(hparamsss)))
                print('Num. of candidate models: %d' % len(hparamss))

            # Causality inference
            n_corrects = _test_find_best_model_main(
                test_params, hparamss, show_progress=show_progress, 
                sampling_mode=sampling_mode)

            # Append result to table
            n_confounders_.append(test_params['gen_data_params'].n_confounders)
            prior_indvdl_.append(hparamss[0]['prior_indvdl'])
            dist_noise_.append(hparamss[0]['dist_noise'])
            n_corrects_.append(n_corrects)

            if show_progress:
                print('Elapsed time: %.1f [sec]\n' % (
                    time.time() - t_start_local))

    # ---- Program finished ----
    print('Program finished at %s' % time.strftime("%c"))
    print('Elapsed time: %.1f [sec]\n' % (time.time() - t_start))

    if show_results:
        df = pd.DataFrame({
            'n_confounders': n_confounders_, 
            'prior_indvdl': prior_indvdl_, 
            'dist_noise': dist_noise_, 
            'n_corrects': n_corrects_, 
        })
        
        return df
    else:
        return None

def _test_find_best_model_main(
    test_params, hparamss, show_progress, sampling_mode):
    n_trials = test_params['n_trials']
    min_corrects = test_params['min_corrects']
    gen_data_params = test_params['gen_data_params']
    
    # ---- Do test ----
    n_corrects = _eval_bmlingam(
        n_trials, gen_data_params, hparamss, show_progress=show_progress, 
        sampling_mode=sampling_mode)
    
    assert(min_corrects <= n_corrects)

    return n_corrects

def _eval_bmlingam(
    n_trials, gen_data_params, hparamss, show_progress, sampling_mode):
    """Evaluate BMLiNGAM model selection. 
    """
    # ---- Loop over trials ----
    n_corrects = 0
    for t in range(n_trials):
        # ---- Generate samples ----
        gen_data_params.seed = t
        data = gen_samples(gen_data_params)
        xs = data['xs']
        causality_true = data['causality_true']

        # ---- Estimate causality ----
        results = find_best_model(xs, hparamss, sampling_mode)
        hparams_best = results[0]
        posterior = results[1]
        causality_est = hparams_best['causality']

        # ---- Check result ----
        if causality_est == causality_true:
            n_corrects += 1

        if show_progress:
            print('causality (true/pred), p(M_MAP|D): (%s/%s), %e' % 
                  (str(causality_true), str(causality_est), posterior))

    return n_corrects
