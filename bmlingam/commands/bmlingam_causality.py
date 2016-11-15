# -*- coding: utf-8 -*-

"""Include functions used in command bin/bmlingam-causality. 
"""
# Author: Taku Yoshioka
# License: MIT

from itertools import combinations
import numpy as np
from os import sep
import pandas as pd
from argparse import ArgumentParser

from bmlingam import load_data, define_hparam_searchspace, InferParams, \
                     infer_causality, save_pklz
from bmlingam.utils import eval_

def _get_pairs(n_variables):
    """Return pairs of indices without repetitions. 
    """
    return list(combinations(range(n_variables), 2))

def _make_table_causal(results):
    """Aggregate inferred causalities included in given results.
    """
    causalities = [r['causality_str'] for r in results]
    post_probs = [r['post_prob'] for r in results]
    post_probs_rev = [r['post_prob_rev'] for r in results]

    return pd.DataFrame({
        'Inferred Causality': causalities, 
        'Posterior probability': post_probs, 
        'Posterior probability of reverse model': post_probs_rev
    })

def _get_optmodel_file(result, result_dir):
    """Returns optimal model filename.
    """
    x1_name = result['x1_name']
    x2_name = result['x2_name']
    return result_dir + sep + '%s_%s.bmlm.pklz' % (x1_name, x2_name)

def parse_args_bmlingam_causality(args=None):
    parser = ArgumentParser()

    # Positional arguments
    parser.add_argument('csv_file', type=str, 
                        help='CSV file of data.')

    # Optional arguments: execution parameters
    parser.add_argument('--result_dir', type=str, 
                        default='.' + sep, 
                        help="""Directory where result files are saved.
                                Default is current directory. """)

    parser.add_argument('--out_optmodelfile', 
                        dest='is_out_optmodelfile', action='store_true', 
                        help="""If this option is choosen (default), optimal model 
                                files will be created.""")
    parser.add_argument('--no_out_optmodelfile', 
                        dest='is_out_optmodelfile', action='store_false', 
                        help="""If this option is choosen, optimal model files 
                                will not be created.""")
    parser.set_defaults(is_out_optmodelfile=True)

    parser.add_argument('--col_names', 
                        default=None, type=str, 
                        help="""Names of column, specified as 'name1,name2,...'
                                (space not allowed). 
                                If set this value 'auto', column names will be 
                                automatically determined as 'x0,x1,...'. 
                                If csv file have column names (the 1st row of the 
                                file), they will be overwritten.
                                """)

    parser.add_argument('--optmodel_files', 
                        default=None, 
                        help="""Filenames of optimal model files. This should 
                                be specified as Python list, e.g., 
                                '["file1", "file2", ...]'. The length of the 
                                list must be the same with the number of all 
                                variable pairs in the data. If None 
                                (default), the filenames are automatically 
                                determined. This parameter overwrites 
                                --out_optmodelfile.""")

    # Get default setting
    default = InferParams()

    # For bool args
    str_bool = lambda b1, b2: ' (default)' if b1 == b2 else ''

    # Optional arguments: inference parameters
    parser.add_argument('--seed', 
                        default=default.seed, type=int, 
                        help="""Specify the seed of random number generator used in 
                                MC sampling. Default is {}. 
                            """.format(default.seed))

    parser.add_argument('--standardize_on', 
                        dest='standardize', action='store_true', 
                        help="""If this option is choosen {}, data is standardized
                                to mean 0 and variance 1 before causal inference. 
                            """.format(str_bool(default.standardize, True)))
    parser.add_argument('--standardize_off', 
                        dest='standardize', action='store_false', 
                        help="""If this option is choosen{}, data is not 
                                standardized.
                            """.format(str_bool(default.standardize, False)))
    parser.set_defaults(standardize=default.standardize)

    parser.add_argument('--fix_mu_zero_on', 
                        dest='fix_mu_zero', action='store_true', 
                        help="""If this option is choosen{}, common interception 
                                parameter mu_1,2 will be treated as 0 
                                (constant), not estimated. 
                            """.format(str_bool(default.fix_mu_zero, True)))
    parser.add_argument('--fix_mu_zero_off', 
                        dest='fix_mu_zero', action='store_false', 
                        help="""If this option is choosen, common 
                                interception parameter mu_1,2 will be included
                                in models as stochastic variables.
                            """.format(str_bool(default.fix_mu_zero, False)))
    parser.set_defaults(fix_mu_zero=default.fix_mu_zero)

    parser.add_argument('--max_c', 
                        default=default.max_c, type=float, 
                        help="""Scale constant on tau_cmmn. Default is {}.
                            """.format(default.max_c))

    parser.add_argument('--n_mc_samples', 
                        default=default.n_mc_samples, type=int, 
                        help="""The number of Monte Carlo sampling in calculation 
                                of marginal likelihood values of models. 
                                Default is {}. 
                            """.format(default.n_mc_samples))

    parser.add_argument('--dist_noise', 
                        default=default.dist_noise, type=str, 
                        help="""Noise distribution. 'laplace' or 'gg' 
                                (Generalized Gaussian). Default is {}.
                            """.format(default.dist_noise))

    parser.add_argument('--df_indvdl', 
                        default=default.df_indvdl, type=float, 
                        help="""Degrees of freedom of T distribution for 
                                the prior of individual specific effects.
                                Default is {}.
                            """.format(default.df_indvdl))

    parser.add_argument('--prior_scale', 
                        default=default.prior_scale, type=str, 
                        help="""Prior distribution on noise variance.
                                'log_normal' or 'tr_normal' 
                                (truncated normal distribution). 
                                Default is {}.
                            """.format(default.prior_scale))

    parser.add_argument('--prior_indvdls', 
                        default=default.prior_indvdls[0], type=str, 
                        help="""Distribution of individual effects in the model. 
                                This argument can be 't', 'gauss' or 'gg'. 
                                If you want to include multiple distributions, 
                                set this argument as 't,gauss', then the 
                                program will apply both of t and Gaussian 
                                distributions to candidate models. 
                                Default is {}.
                            """.format(default.prior_indvdls[0]))

    parser.add_argument('--cs', 
                        default='0,.2,.4,.6,.8', type=str, 
                        help="""Scales of stds of the individual specific effects. 
                                Default is '0,.2,.4,.6,.8'. """)

    parser.add_argument('--L_cov_21s', type=str, 
                        default='[-0.9,-0.7,-0.5,-0.3,0,.3,.5,.7,.9]', 
                        help="""List of correlations of individual specific 
                                effects. 
                                Default is 
                                '[-0.9,-0.7,-0.5,-0.3,0,.3,.5,.7,.9]'. """)

    parser.add_argument('--betas_indvdl', 
                        default='.25,.5,.75,1.', type=str, 
                        help="""Shape parameter values of generalized Gaussian 
                                distributions for individual specific effects. 
                                When prior_indvdls includes 'gg', all of the 
                                beta values will be tested. .5 and 1. correspond 
                                to Laplace and Gaussian distributions, 
                                respectively. 
                                Default is '.25,.5,.75,1.'. """)

    parser.add_argument('--betas_noise', 
                        default='.25,.5,.75,1.', type=str, 
                        help="""Shape parameter values of generalized Gaussian 
                                distributions for observation noise. 
                                When dist_noise includes 'gg', all of the 
                                beta values will be tested. .5 and 1. correspond 
                                to Laplace and Gaussian distributions, 
                                respectively. 
                                Default is '.25,.5,.75,1.'. """)

    parser.add_argument('--causalities', 
                        default='x1->x2, x2->x1', type=str, 
                        help="""Causalities to be tested. If set to 'x1->x2' or 
                                'x2->x1', causality is not inferred and other 
                                hyperparameters are searched. 
                                Default is 'x1->x2, x2->x1'. """)

    parser.add_argument('--sampling_mode', 
                        default='cache', type=str, 
                        help="""Specify sampling mode for numerical integration 
                                via MC. Options are 'normal', 'cache', 'cache_mp2', 
                                'cache_mp4' or 'cache_mp8'. 'normal' means naive 
                                MC sampling: generate random values at each 
                                hyperparameter set. When specified 'chache', 
                                random values are generated only at the beginning of 
                                the program and applied to marginal likelihood 
                                calculation with difference hyperparameter sets. 
                                Multiprocessing is supported with the option 
                                'cache_mp[2, 4, 8]', using 2, 4 or 8 cores.""")

    args_ = parser.parse_args(args)

    return {
        'csv_file': args_.csv_file, 
        'result_dir': args_.result_dir, 
        'is_out_optmodelfile': args_.is_out_optmodelfile, 
        'col_names': None if args_.col_names is None else args_.col_names.split(','), 
        'infer_params': InferParams(
            seed = args_.seed, 
            standardize = args_.standardize, 
            fix_mu_zero = args_.fix_mu_zero, 
            max_c = args_.max_c, 
            n_mc_samples = args_.n_mc_samples, 
            P_M1 = 0.5, 
            P_M2 = 0.5, 
            dist_noise = args_.dist_noise, 
            df_indvdl = args_.df_indvdl,
            prior_scale = args_.prior_scale,  
            prior_indvdls = args_.prior_indvdls.split(','), 
            cs = np.array(args_.cs.split(',')).astype(float), 
            L_cov_21s = eval_(args_.L_cov_21s), 
            betas_indvdl = np.array(args_.betas_indvdl.split(',')).astype(float), 
            betas_noise = np.array(args_.betas_noise.split(',')).astype(float), 
            sampling_mode = args_.sampling_mode
        ), 
        'optmodel_files': args_.optmodel_files
    }

def bmlingam_causality(
    csv_file, result_dir, is_out_optmodelfile, col_names, infer_params, 
    optmodel_files):
    """Infer causality of all pairs in the data. 
    """
    assert(type(infer_params) == InferParams)

    if type(optmodel_files) is str:
        optmodel_files = [optmodel_files]

    print('---- Algorithm parameters ----')
    print('Number of MC samples: %d' % infer_params.n_mc_samples)
    hparamss = define_hparam_searchspace(infer_params)
    print('Number of candidate models: %d' % len(hparamss))
    print('')

    # Load data and infer causality
    df = load_data(csv_file, col_names) # Pandas dataframe

    # Get all possible pairs of variables
    pairs = _get_pairs(len(df.columns))

    # Check optimal model files
    if optmodel_files is not None:
        assert(len(optmodel_files) == len(pairs))
        optmodel_files_ = optmodel_files

    # Infer causality over all variable pairs
    data = df.as_matrix()
    varnames = df.columns.values
    results = [infer_causality(data[:, pair], infer_params, 
               varnames[list(pair)]) for pair in pairs]

    # Summarize inference
    table_causal = _make_table_causal(results)

    # Set optimal model files
    if optmodel_files is None:
        if result_dir is not None:
            optmodel_files_ = [_get_optmodel_file(result, result_dir)
                               for result in results]
        else:
            optmodel_files_ = []

    # Conditions to save results (and optimal models)
    cond_save_results = (result_dir is not None) and (0 < len(result_dir))
    cond_save_optmodels = 0 < len(optmodel_files_) and is_out_optmodelfile

    # Save results
    if cond_save_results:
        result_file = result_dir + sep + 'causality.csv'
        table_causal.to_csv(result_file)
        print('Inferred causality table was saved as %s.' % result_file)

    # Save optimal models
    if cond_save_optmodels:
        for result, optmodel_file in zip(results, optmodel_files_):
            save_pklz(optmodel_file, result)
            print('Optimal model was saved as %s.' % optmodel_file)
