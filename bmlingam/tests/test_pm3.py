# -*- coding: utf-8 -*-

"""Test MCMC sampling.
"""
# Author: Taku Yoshioka
# License: MIT

# ---- Public modules ----
import matplotlib.pyplot as plt
import numpy as np
import time

# ---- bmlingam modules ----
from bmlingam import define_hparam_searchspace, MCMCParams, \
                     do_mcmc_bmlingam, InferParams, find_best_model
from bmlingam.tests.gendata import gen_samples, GenDataParams
from bmlingam.progressbar import ProgressBar

"""Program of testing accuracy of posterior inference
"""

infer_params_default = InferParams(
    seed=0, 
    standardize=False, 
    subtract_mu_reg=False, 
    fix_mu_zero=False, 
    prior_var_mu='auto', 
    max_c=10.0, 
    n_mc_samples=1000, 
    dist_noise='laplace', 
    df_indvdl=6.0, 
    prior_scale='tr_normal', 
    prior_indvdls=['t'], 
    L_cov_21s=[-.9, -.7, -.5, -.3, 0, .3, .5, .7, .9], 
    betas_indvdl=None, # [0.25, 0.5, 0.75, 1.], 
    betas_noise=None, # [0.25, 0.5, 0.75, 1.], 
    causalities=[[1, 2], [2, 1]], 
)

mcmc_params_default = MCMCParams(
    n_burn=10000, 
    n_mcmc_samples=10000
)

gen_data_params_default = GenDataParams(
    n_confounders=6, 
    n_samples=100, 
    sample_coef='r2intervals', 
    dists_conf=['all'], 
    mu1_dist='randn', 
    mu2_dist='randn', 
    e1_dists=['all'], 
    e2_dists=['all'],
    e1_std='r2intervals', 
    e2_std='r2intervals', 
    f1_coef_dists='r2intervals', 
    f2_coef_dists='r2intervals', 
    seed=0
)

def _gen_data(i, gen_data_params):
    # Generate artifitial data
    if type(gen_data_params) is GenDataParams:
        gen_data_params.seed = i * 1000
        data = gen_samples(gen_data_params)
    else: # Old program
        import bmlingam.tests.gendata_old as old
        data = old.gen_samples(
            n_confounders=gen_data_params['n_confounders'], 
            n_samples=gen_data_params['n_samples'], 
            rng=np.random.RandomState(i)
        )

    return data

def _estimate_hparams(xs, infer_params):
    assert(type(infer_params) == InferParams)

    sampling_mode = infer_params.sampling_mode
    hparamss = define_hparam_searchspace(infer_params)
    results = find_best_model(xs, hparamss, sampling_mode)
    hparams_best = results[0]
    bf = results[2] - results[5] # Bayes factor
    
    return hparams_best, bf

def _mainloop(n_trials, gen_data_params, infer_params, mcmc_params):
    # ---- Loop over trials ----
    bss, causalities, sampless = [], [], []
    ts_causal_inference = []
    ts_posterior_inference = []

    # Main loop
    p = ProgressBar(n_trials)
    print('Loop over trials')
    for i in range(n_trials):
        p.animate(i)

        # Generate artificial data
        data = _gen_data(i, gen_data_params)
        b_true = data['b']
        causality_true = data['causality_true']
        sampless.append(data)

        # Causal inference based on Bayesian model selection
        t = time.time()
        hparams, bf = _estimate_hparams(data['xs'], infer_params)
        causality_pred = hparams['causality']
        ts_causal_inference.append(time.time() - t); del t

        # Draw samples of regression coefficient from posterior with MCMC
        t = time.time()
        if mcmc_params.n_mcmc_samples == 0:
            b_post = 0.
        else:
            trace = do_mcmc_bmlingam(
                data['xs'], hparams, mcmc_params)
            b_post = np.mean(trace['b'])
        ts_posterior_inference.append(time.time() - t); del t

        # Store results
        bss.append(np.array([b_true, b_post]))
        causalities.append((causality_true, causality_pred))
        print('debug', causality_true, causality_pred, bf)
    p.animate(i)
    print('\n')

    return {
        'bss': bss, # True and estimated regression coefficient of the model
        'causalities': causalities, # True and estimated causalities
        'sampless': sampless, # Artificial data, 
        'ts_causal_inference': ts_causal_inference, 
        'ts_posterior_inference': ts_posterior_inference, 
    }

def test_posterior_inference(
    infer_params=None, mcmc_params=None, gen_data_params=None, 
    n_trials=50, plot_result=False, plot_samples_all=False, show_result=False, 
    show_result_all=False):
    """Test accuracy of posterior inference of coefficient of bmlingam. 

    This function is invoked from bmlingam.tests.test_bmlingam.test_bmlingam().  

    :param infer_params: Inference parameters. 
    :type infer_params: bmlingam.hparam.InferParams
    :param mcmc_params: MCMC parameters. 
    :type mcmc_params: bmlingam.bmlingam_pm3.MCMCParams
    :param gen_data_params: Data generation parameters. 
    :type gen_data_params: bmlingam.tests.gendata.GenDataParams

    If these are set to None, default values are used. 

    :param str sampling_mode: MonteCarlo sampling mode.

    The following options are supported:

    - 'normal': Based on naive implementation. 
    - 'cache': Using cache of random variables. 
    - 'cache_mp2': Using cache on 2 processes. 
    - 'cache_mp4': Using cache on 4 processes. 
    - 'cache_mp8': Using cache on 8 processes. 
    """
    t_start = time.time()

    # Set default parameters
    if infer_params is None: infer_params = infer_params_default
    if mcmc_params is None: mcmc_params = mcmc_params_default
    if gen_data_params is None: gen_data_params = gen_data_params_default

    # Perform evaluation
    results = _mainloop(n_trials, gen_data_params, infer_params, mcmc_params)

    # Show results
    if show_result:
        # Correct rate of causality
        n_corrects = len([1 for cs in results['causalities'] if cs[0] == cs[1]])
        r = n_corrects / float(n_trials)
        print('Correct rate of causality estimation = %.3f' % r)
        print('Average time for causal inference = {:.1f}'.format(
            np.mean(results['ts_causal_inference'])))
        print('Averate time for coef posterior inference = {:.1f}\n'.format(
            np.mean(results['ts_posterior_inference'])))

    if plot_samples_all:
        for cs, samples in zip(results['causalities'], results['sampless']):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            xs = samples['xs']
            ax.scatter(xs[:, 0], xs[:, 1])
            ax.set_title('True: %s, Pred: %s' % (str(cs[0]), str(cs[1])))
            ax.set_aspect('equal', 'datalim')

    if show_result_all:
        print('---- List of causalities (true, pred) ----')
        for cs in results['causalities']:
            print(cs[0], cs[1])
        print('')

    if plot_result:
        # Scatter plot
        bs_true = np.vstack(results['bss'])[:, 0]
        bs_post = np.vstack(results['bss'])[:, 1]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(bs_true, bs_post)
        ax.set_xlabel('True')
        ax.set_ylabel('Posterior mean')
        ax.set_title('Estimated coefficients')

    # ---- Program finished ----
    print('Program finished at %s' % time.strftime("%c"))
    print('Elapsed time: %.1f [sec]\n' % (time.time() - t_start))
