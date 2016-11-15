# -*- coding: utf-8 -*-

"""Test Bayesian mixed LiNGAM model.
"""
# Author: Taku Yoshioka, Shohei Shimizu
# License: MIT

# ---- Public modules ----
from nose.tools import ok_
import numpy as np
import time

# ---- lingam modules ----
from bmlingam import bayesmixedlingam_np, bayesmixedlingam_ref, \
                     define_hparam_searchspace, comp_logP_bmlingam_np, \
                     comp_logP_bmlingam_pm3, find_best_model
from bmlingam.tests.gendata import gen_samples, gendata_latents

"""Equivalence of np and pymc implementations
"""

def test_sampling(n_mc_samples=100):
    raise NotImplementedError('This function is not implemented.')

    rng = np.random.RandomState(0)

    # ---- Generate samples ----
    data = gen_samples(n_confounders=1, n_samples=100, rng=rng)
    xs = data['xs']
    causality_true = data['causality_true']

    # ---- Get a hyperparameter set ----
    hparamss = define_hparam_searchspace(n_mc_samples=n_mc_samples)
    hparams = hparamss[200]

    # ---- MC sampling ----
    logp_np, traces_np = comp_logP_bmlingam_np(xs, hparams, rng)
    logp_pm2, traces_pm2 = comp_logP_bmlingam_pm2(xs, hparams, rng)
    # logp_pm3, traces_pm3 = comp_logP_bmlingam_pm3(xs, hparams, rng)

    return {
        'logp_np': logp_np, 
        'traces_np': traces_np, 
        'logp_pm2': logp_pm2, 
        'traces_pm2': traces_pm2, 
        # 'logp_pm3': logp_pm3, 
        # 'traces_pm3': traces_pm3, 
    }



"""Accuracy of model selection (causality inference)
"""

def test_bmlingam_np(
    show_result=False, tied_sampling=False, 
    n_confounderss=[0, 1, 6, 12], n_trials=10, min_corrects=[6, 6, 6, 6], 
    n_samples=100, max_c=10., n_mc_samples=1000, normalize_samples=False, 
    prior_indvdls=['t'], assertive=True):
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
    test_params = {
        'n_confounderss': n_confounderss, 
        'n_trials': n_trials, 
        'min_corrects': min_corrects, 
        'n_samples': n_samples, 
        'max_c': max_c, 
        'n_mc_samples': n_mc_samples, 
        'normalize_samples': normalize_samples, 
        'prior_indvdls': prior_indvdls
    }
    comp_logP_func = comp_logP_bmlingam_np
    _test_bmlingam_main(
        comp_logP_func, test_params, show_result, tied_sampling, assertive)

def test_bmlingam_pymc(
    show_result=False, 
    n_confounderss=[0, 1, 6, 12], n_trials=10, min_corrects=[6, 6, 6, 6], 
    n_samples=100, max_c=10., n_mc_samples=1000, normalize_samples=False, 
    prior_indvdls=['t'], assertive=True):
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
    """
    raise NotImplementedError(
        """This function is not correctly implemented. 
        Need to numerically calculate harmonic mean.
        """)
    test_params = {
        'n_confounderss': n_confounderss, 
        'n_trials': n_trials, 
        'min_corrects': min_corrects, 
        'n_samples': n_samples, 
        'max_c': max_c, 
        'n_mc_samples': n_mc_samples, 
        'normalize_samples': normalize_samples, 
        'prior_indvdls': prior_indvdls
    }
    comp_logP_func = comp_logP_bmlingam_pymc
    _test_bmlingam_main(comp_logP_func, test_params, show_result)

def _test_bmlingam_main(
    comp_logP_func, test_params, show_result=False, tied_sampling=False, 
    assertive=True):
    """Test estimation using Bayesian mixed LiNGAM model.

    This function is invoked from test_bmlingam_np() and test_bmlingam_pymc(). 
    """
    t_start = time.time()

    # ---- Testing parameters ----
    n_confounderss = test_params['n_confounderss']
    n_trials = test_params['n_trials']
    min_corrects = test_params['min_corrects']
    n_samples = test_params['n_samples']
    max_c = test_params['max_c']
    n_mc_samples = test_params['n_mc_samples']
    normalize_samples = test_params['normalize_samples']
    prior_indvdls = test_params['prior_indvdls']
    
    # ---- Hyperparameter search space ----
    hparamss = define_hparam_searchspace(
        tied_sampling=tied_sampling, max_c=max_c, n_mc_samples=n_mc_samples, 
        prior_indvdls=prior_indvdls)

    # ---- Do test ----
    rng = np.random.RandomState(0)
    for i in xrange(len(n_confounderss)):
        n_corrects = _eval_bmlingam(
            comp_logP_func, n_confounderss[i], n_trials, n_samples, hparamss, 
            rng, show_result=show_result, tied_sampling=tied_sampling, 
            normalize_samples=normalize_samples)
        
        if show_result:
            print(('n_confounders=%d, %d correct inferences ' + 
                   'out of 10 trials') % (n_confounderss[i], n_corrects))

        if assertive:
            ok_(min_corrects[i] <= n_corrects)

    if show_result:
        print('')
        print('Program finished at %s' % time.strftime("%c"))
        print('Elapsed time: %.1f [sec]' % (time.time() - t_start))
        print('')

    return

def _eval_bmlingam(
    comp_logP_func, n_confounders, n_trials, n_samples, hparamss, rng, 
    show_result, tied_sampling=False, normalize_samples=False):
    """Evaluate BMLiNGAM model selection. 

    This function is invoked from _test_bmlingam_main(). 
    """
    # ---- Loop over trials ----
    n_corrects = 0
    for t in xrange(n_trials):
        # ---- Generate samples ----
        data = gen_samples(
            n_confounders, n_samples, rng, normalize_samples=normalize_samples)
        xs = data['xs']
        causality_true = data['causality_true']

        # ---- Estimate causality ----
        if tied_sampling: 
            causality_est, _ = bayesmixedlingam_np(xs, hparamss, rng)
            posterior = np.nan
        else:
            hparams_best, posterior = find_best_model(xs, hparamss, rng)
            causality_est = hparams_best['causality']

        # ---- Check result ----
        if causality_est == causality_true:
            n_corrects += 1

        if show_result:
            print('causality (true/pred), p(M_MAP|D): (%s/%s), %e' % 
                  (str(causality_true), str(causality_est), posterior))

    return n_corrects



"""Reference implementation
"""

def test_estimation_t_ref(show_result=False):
    """Test estimation using Bayesian mixed LiNGAM model (student-t).

    The tests run over numbers of confounders: 0, 1, 6 and 12. 
    Each of the tests passes if the ratio of correct estimations is 
    greater than a threshold for each of settings. 

    The following code is testing parameters:

    .. code:: python

        prior_indvdl = 't' # Distribution of individual specific effects
        n_confounderss = [0, 1, 6, 12] # Number of confounders
        n_trials = 10 # Number of trials (inferences)
        min_corrects = [6, 6, 6, 6] # Threshold of correct inferences
        n_samples = 100 # Number of samples in every trial
        n_theta_sampling = 1000 # Monte calro samples
        P_M1, P_M2 = .5, .5 # Prior prob of the two models

    :param bool show_result: If true, the number of correct trials in \
        estimation is printed. 
    """
    # ---- Testing parameters ----
    prior_indvdl = 't'
    n_confounderss = [0, 1, 6, 12]
    n_trials = 10
    min_corrects = [6, 6, 6, 6]
    n_samples = 100
    n_theta_sampling = 1000
    P_M1, P_M2 = .5, .5
    rng = np.random.RandomState(0)

    # ---- Do test ----
    for i in xrange(len(n_confounderss)):
        n_corrects = _eval_bayesmixedlingam_ref(
            n_confounderss[i], n_trials, n_samples, n_theta_sampling, 
            P_M1, P_M2, prior_indvdl, rng)

        if show_result:
            print(('n_confounders=%d, %d correct inferences ' + 
                   'out of 10 trials') % (n_confounderss[i], n_corrects))

        ok_(min_corrects[i] <= n_corrects)

    return

def test_estimation_gauss_ref(show_result=False):
    """Test estimation using Bayesian mixed LiNGAM model (gauss).

    The tests run over numbers of confounders: 0, 1, 6 and 12. 
    Each of the tests passes if the ratio of correct estimations is 
    greater than a threshold for each of settings. 

    The following code is testing parameters:

    .. code:: python

        prior_indvdl = 'gauss' # Distribution of individual specific effects
        n_confounderss = [0, 1, 6, 12] # Number of confounders
        n_trials = 10 # Number of trials (inferences)
        min_corrects = [6, 6, 6, 6] # Threshold of correct inferences
        n_samples = 100 # Number of samples in every trial
        n_theta_sampling = 1000 # Monte calro samples
        P_M1, P_M2 = .5, .5 # Prior prob of the two models

    :param bool show_result: If true, the number of correct trials in \
        estimation is printed. 
    """
    # ---- Testing parameters ----
    prior_indvdl = 'gauss'
    n_confounderss = [0, 1, 6, 12]
    n_trials = 10
    min_corrects = [6, 6, 6, 6]
    n_samples = 100
    n_theta_sampling = 1000
    P_M1, P_M2 = .5, .5
    rng = np.random.RandomState(0)

    # ---- Do test ----
    for i in xrange(len(n_confounderss)):
        n_corrects = _eval_bayesmixedlingam_ref(
            n_confounderss[i], n_trials, n_samples, n_theta_sampling, 
            P_M1, P_M2, prior_indvdl, rng)
        ok_(min_corrects[i] <= n_corrects)

        if show_result:
            print(('n_confounders=%d, %d correct inferences ' + 
                   'out of 10 trials') % (n_confounderss[i], n_corrects))

    return

def _eval_bayesmixedlingam_ref(
    n_confounders, n_trials, n_samples, n_theta_sampling, P_M1, P_M2, 
    prior_indvdl, rng):
    """Used in test_estimation(). 
    """
    # ---- Loop over trials ----
    n_corrects = 0
    for t in xrange(n_trials):
        # ---- Generate samples ----
        xs = gendata_latents(n_confounders, n_samples, rng)
        if rng.randn(1) < 0:
            xs_ = np.vstack((xs[:, 1], xs[:, 0])).T
            causality_true = [2, 1]
        else:
            xs_ = xs
            causality_true = [1, 2]

        # ---- Estimate causality ----
        kest, _ = bayesmixedlingam_ref(
            xs_, n_theta_sampling, P_M1, P_M2, prior_indvdl, rng)

        # ---- Check result ----
        if kest == causality_true:
            n_corrects += 1

    return n_corrects

"""Equivalence of log marginal likelihood of ref and np impls
"""

def test_equiv_logps_ref_np(n_theta_sampling=1000):
    """Check equivalence of log marinal likelihood of ref and np impls.
    """
    n_confounders = 6
    n_samples = 100
    rng = np.random.RandomState(0)

    hparamss = define_hparam_searchspace(
        n_theta_sampling=n_theta_sampling, prior_indvdls=['t'])

    n_theta_sampling = hparamss[0]['n_theta_sampling']
    P_M1 = hparamss[0]['P_M1']
    P_M2 = hparamss[0]['P_M2']
    prior_indvdl = hparamss[0]['prior_indvdl']

    # ---- Generate samples ----
    data = _gen_samples(n_confounders, n_samples, rng)
    xs = data['xs']
    causality_true = data['causality_true']

    # ---- Inference ----
    _, logPs_ref = bayesmixedlingam_ref(
            xs, n_theta_sampling, P_M1, P_M2, prior_indvdl, rng)
    print(logPs_ref)
    _, logPs_np = bayesmixedlingam_np(xs, hparamss, rng)
    print(logPs_np)

"""Deprecated
"""

def test_estimation_bmlingam_t_np(show_result=False, tied_sampling=False):
    u"""
    TODO: If this function is invoked by automatic testing suites (e.g., nose), 
          it should be skipped. But I dont know how to do that. In future, 
          this function will be deleted. 
    """
    raise DeprecationWarning(
        'test_estimation_bmlingam_t_np() is deprecated.' +
        'Use test_bmlingam_np() instead.'
    )
