# -*- coding: utf-8 -*-

# Author: Taku Yoshioka, Shohei Shimizu
# License: MIT

from bmlingam import define_hparam_searchspace, find_best_model, InferParams
from bmlingam.utils import gen_artificial_data, GenDataParams

def infer_params1():
    return InferParams(
        sampling_mode='cache_mp4', 
        standardize=True, 
        fix_mu_zero=True, 
        max_c=1.0, 
        n_mc_samples=10000, 
        prior_scale='uniform', 
        L_cov_21s=[-0.8, -0.6, -0.4, 0.4, 0.6, 0.8], 
        scale_coeff=2.0 / 3.0, 
        cs=[0.4, 0.6, 0.8]
    )

def infer_params2():
    infer_params = infer_params1()
    infer_params.L_cov_21s = ['U(-0.99, 0.99)']

    return infer_params

def test_find_best_model(verbose=False):
    gen_data_params = GenDataParams(
        n_samples=200, 
        mu1_dist=5.0, 
        mu2_dist=10.0, 
        f1_coef=[1.0, 1.0, 1.5], 
        f2_coef=[1.0, 2.0, 0.5], 
        conf_dist=[['laplace'], ['exp'], ['uniform']], 
        e1_dist=['laplace'], 
        e2_dist=['laplace'], 
        e1_std=3.0, 
        e2_std=3.0, 
        fix_causality=False, 
        seed=0
    )

    # gen_data_params = deepcopy(gen_data_params_default)
    gen_data_params.n_samples = 200
    gen_data_params.n_confounders = 3
    gen_data_params.dists_e1 = ['laplace']
    gen_data_params.dists_e2 = ['laplace']
    gen_data_params.dist_be1 = 'be1=9.0'
    gen_data_params.dist_be2 = 'be2=9.0'
    gen_data_params.dist_bf1s = '1., 1., 1.5'
    gen_data_params.dist_bf2s = '1., 2., 0.5'
    gen_data_params.dists_conf = [['laplace'], ['exp'], ['uniform']]
    gen_data_params.dist_mu1 = 'mu1=5.0'
    gen_data_params.dist_mu2 = 'mu2=10.0'

    data = gen_artificial_data(gen_data_params)
    xs = data['xs']

    infer_params = infer_params1()
    sampling_mode = infer_params.sampling_mode
    hparamss = define_hparam_searchspace(infer_params)
    result1 = find_best_model(xs, hparamss, sampling_mode)
    print(result1)

    infer_params = infer_params2()
    sampling_mode = infer_params.sampling_mode
    hparamss = define_hparam_searchspace(infer_params)
    result2 = find_best_model(xs, hparamss, sampling_mode)
    print(result2)
