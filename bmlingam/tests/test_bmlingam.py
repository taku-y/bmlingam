# -*- coding: utf-8 -*-

u"""Program for testing bmlingam.
"""
# Author: Taku Yoshioka
# License: MIT

from copy import deepcopy

import six

from bmlingam.tests.test_pm3 import test_posterior_inference, \
                                    infer_params_default, \
                                    mcmc_params_default, \
                                    gen_data_params_default

def _wrap_list(a):
    if isinstance(a, (set, list)):
        return a
    else:
        return [a]

def _get_test_conditions():
    """Get a list of test conditions. 

    Each test condition consists of parameters for causal inference, 
    estimation of regression coefficient via MCMC, and artificial data 
    generation. 
    """
    infer_paramss = _get_infer_paramss()
    mcmc_paramss = _get_mcmc_paramss()
    gen_data_paramss = _get_gen_data_paramss()

    test_conds = [
        {
            'infer_params_name': infer_params_name, 
            'mcmc_params_name': mcmc_params_name, 
            'gen_data_params_name': gen_data_params_name, 
            'infer_params': infer_params, 
            'mcmc_params': mcmc_params, 
            'gen_data_params': gen_data_params, 
        }
        for infer_params_name, infer_params in six.iteritems(infer_paramss)
        for mcmc_params_name, mcmc_params in six.iteritems(mcmc_paramss)
        for gen_data_params_name, gen_data_params in six.iteritems(gen_data_paramss)
    ]

    return test_conds

def _get_infer_paramss():
    infer_params_mp4 = deepcopy(infer_params_default)
    infer_params_mp4.sampling_mode = 'cache_mp4'

    infer_params_standardize = deepcopy(infer_params_mp4)
    infer_params_standardize.standardize = True

    infer_params_keepratio = deepcopy(infer_params_mp4)
    infer_params_keepratio.standardize = 'keepratio'

    infer_params_scaling = deepcopy(infer_params_mp4)
    infer_params_scaling.standardize = 'scaling'

    infer_params_commonscaling = deepcopy(infer_params_mp4)
    infer_params_commonscaling.standardize = 'commonscaling'

    return {
        'default': infer_params_default, 
        'cache_mp4': infer_params_mp4, 
        'standardize': infer_params_standardize, 
        'keepratio': infer_params_keepratio, 
        'scaling': infer_params_scaling, 
        'commonscaling': infer_params_commonscaling
    }

def _get_mcmc_paramss():
    return {
        'default': mcmc_params_default
    }

def _get_gen_data_paramss():
    confounders0 = deepcopy(gen_data_params_default)
    confounders0.n_confounders = 0
    confounders1 = deepcopy(gen_data_params_default)
    confounders1.n_confounders = 1
    confounders12 = deepcopy(gen_data_params_default)
    confounders12.n_confounders = 12

    return {
        'default': gen_data_params_default, 
        'confounders0': confounders0, 
        'confounders1': confounders1, 
        'confounders12': confounders12, 
    }

def test_bmlingam(
    infer_params_names='default', mcmc_params_names='default', 
    gen_data_params_names='default', n_trials=50, plot_result=False, 
    show_result=True, show_result_all=False):
    """Test causal inference and estimation of regression coefficient.

    This program is used not only for testing but also for checking 
    accuracy of the inference. 

    Testing condition consists of parameter sets below:

    :param infer_params_names: Name of inference parameters. 
    :type infer_params_names: str or list of str
    :param mcmc_params_names: Name of MCMC parameters. 
    :type mcmc_params_names: str or list of str
    :param gen_data_params_names: Name of data generation parameters. 
    :type gen_data_params_names: str or list of str

    See get_infer_paramss(), get_mcmc_paramss() and get_data_gen_paramss(). 
    """
    infer_paramss = _get_infer_paramss()
    mcmc_paramss = _get_mcmc_paramss()
    gen_data_paramss = _get_gen_data_paramss()

    infer_params_names_ = _wrap_list(infer_params_names)
    mcmc_params_names_ = _wrap_list(mcmc_params_names)
    gen_data_params_names_ = _wrap_list(gen_data_params_names)

    for ip in infer_params_names_:
        infer_params = infer_paramss[ip]
        for mp in mcmc_params_names_:
            mcmc_params = mcmc_paramss[mp]
            for gp in gen_data_params_names_:
                gen_data_params = gen_data_paramss[gp]

                print('Test condition')
                print('  infer_params_name = {}'.format(ip))
                print('  mcmc_params_name = {}'.format(mp))
                print('  gen_data_params_name = {}\n'.format(gp))

                test_posterior_inference(
                    infer_params=infer_params,
                    mcmc_params=mcmc_params,
                    gen_data_params=gen_data_params, 
                    n_trials=n_trials, 
                    plot_result=plot_result, 
                    show_result=show_result, 
                    show_result_all=show_result_all
                )

def test_bmlingam_old(
    infer_params_name='default', mcmc_params_name='default', 
    n_confounderss=[0, 1, 6, 12], n_trials=10, plot_result=True, 
    show_result=True, show_result_all=True):
    """Test inference using old program to make artificial data. 
    """
    infer_paramss = _get_infer_paramss()
    mcmc_paramss = _get_mcmc_paramss()

    print('Test condition')
    print('  infer_params_name = {}'.format(infer_params_name))
    print('  mcmc_params_name = {}'.format(mcmc_params_name))
    print('  gen_data_params_name = None (using old program to make data)\n')

    for n_confounders in n_confounderss:
        gen_data_params = {
            'n_confounders': n_confounders, 
            'n_samples': 100
        }

        test_posterior_inference(
            infer_params=infer_paramss[infer_params_name],
            mcmc_params=mcmc_paramss[mcmc_params_name],
            gen_data_params=gen_data_params, 
            n_trials=n_trials, 
            plot_result=plot_result, 
            show_result=show_result, 
            show_result_all=show_result_all
        )
