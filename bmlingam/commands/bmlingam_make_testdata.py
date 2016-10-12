# -*- coding: utf-8 -*-

"""Include functions used in command bin/bmlingam-make-testdata. 
"""
# Author: Taku Yoshioka
# License: MIT

import argparse
import numpy as np

from bmlingam.utils import eval_, wrap_list
from bmlingam.utils.gendata import gen_artificial_data, GenDataParams

def parse_args_bmlingam_make_testdata(args=None):
    parser = argparse.ArgumentParser()

    # Get default setting
    default = GenDataParams()

    # Optional arguments
    parser.add_argument('--csv_file', type=str, 
                        default='./sampledata.csv', 
                        help='CSV file of data. Default is ./sampledata.csv.')

    parser.add_argument('--n_samples', 
                        default=default.n_samples, type=int, 
                        help="""The number of samples. 
                                See bmlingam.utils.gendata.GenDataParams.
                                Default is %s. """ % (default.n_samples))

    parser.add_argument('--b21_dist', 
                        default=str(default.b21_dist), type=str, 
                        help="""Distribution of :math:`b_{21}`. If `--b21_dist` 
                                is `'r2interval'`, :math:`b21` is drawn from a 
                                mixture of two uniform distributions 
                                :math:`U(-1.5, -0.5)` and :math:`U0.5, 1.5]`, 
                                as appeared in the paper. If a float value is 
                                given, :math:`b_{21}` is set to that value. 
                                Default is %s. """ % str(default.b21_dist))

    parser.add_argument('--mu1_dist', 
                        default=str(default.mu1_dist), type=str, 
                        help="""Distribution of mu1 (common interception). 
                                See bmlingam.utils.gendata.GenDataParams. 
                                Default is %s. """ % str(default.mu1_dist))

    parser.add_argument('--mu2_dist', 
                        default=str(default.mu2_dist), type=str, 
                        help="""Distribution of mu2 (common interception). 
                                See bmlingam.utils.gendata.GenDataParams. 
                                Default is %s. """ % str(default.mu2_dist))

    parser.add_argument('--f1_coef', 
                        default=str(default.f1_coef), type=str, 
                        help="""Std the confound term in the 1st variable. 
                                It is given as Python list whose length is the 
                                number of confounders. The length must be the 
                                same with '--conf_dist'. Example: 
                                '[1.0, 2.0, 'r2intervals']', where two 
                                confounders will be added with constant stds 
                                and the last one randomly scaled following 
                                U([0.5, 1.5]). Default is %s. 
                                """ % str(default.f1_coef))

    parser.add_argument('--f2_coef', 
                        default=str(default.f2_coef), type=str, 
                        help="""Std the confound term in the 1st variable. 
                                It is given as Python list whose length is the 
                                number of confounders. The length must be the 
                                same with '--conf_dist'. Example: 
                                '[1.0, 2.0, 'r2intervals']', where two 
                                confounders will be added with constant stds 
                                and the last one randomly scaled following 
                                U([0.5, 1.5]). Default is %s. 
                                """ % str(default.f2_coef))

    parser.add_argument('--conf_dist', 
                        default=str(default.conf_dist), type=str, 
                        help="""List of possible distributions for confound 
                                factors given as Python list. The length of the 
                                list is equal to the number of confounders. 
                                For example, if '[['laplace'], ['exp', 'uniform']]', 
                                the 1st confounder is distributed according to 
                                the standard Laplace distribution and the 2nd 
                                confounder is drawn from either the exponential 
                                or uniform distributions with 50%% probability. 
                                If 'all', all distributions in usr_distrib() 
                                can be taken. Default is %s. 
                                """ % str(default.conf_dist))

    parser.add_argument('--e1_std', 
                        default=str(default.e1_std), type=str, 
                        help="""Distribution of the standard deviation of 
                                noise e1. It is set to 'r2intervals' 
                                (U(0.5, 1.5)) or float value (constant, note 
                                that standard deviation is constant while noise 
                                is random). Default is %s. 
                                """ % str(default.e1_std))

    parser.add_argument('--e2_std', 
                        default=str(default.e2_std), type=str, 
                        help="""Distribution of the standard deviation of 
                                noise e2. It is set to 'r2intervals' 
                                (U(0.5, 1.5)) or float value (constant, note 
                                that standard deviation is constant while noise 
                                is random). Default is %s. 
                                """ % str(default.e2_std))

    parser.add_argument('--e1_dist', 
                        default=str(default.e1_dist), type=str, 
                        help="""Possible distributions of noise e1. 
                                Default is %s. 
                                """ % str(default.e1_dist))

    parser.add_argument('--e2_dist', 
                        default=str(default.e2_dist), type=str, 
                        help="""Possible distributions of noise e2. 
                                Default is %s. 
                                """ % str(default.e2_dist))

    parser.add_argument('--random_seed', 
                        default=None, type=int, 
                        help="""Seed of the random number generator. 
                                Default is None. """)

    args_ = parser.parse_args(args)

    return {
        'csv_file': args_.csv_file, 
        'gen_data_params': GenDataParams(
            n_samples=args_.n_samples, 
            b21_dist=eval_(args_.b21_dist), 
            mu1_dist=eval_(args_.mu1_dist), 
            mu2_dist=eval_(args_.mu2_dist), 
            f1_scale=1.0, 
            f2_scale=1.0, 
            f1_coef=wrap_list(eval_(args_.f1_coef)),
            f2_coef=wrap_list(eval_(args_.f2_coef)),
            conf_dist=wrap_list(eval_(args_.conf_dist)), 
            e1_std=eval_(args_.e1_std), 
            e2_std=eval_(args_.e2_std),  
            e1_dist=eval_(args_.e1_dist), 
            e2_dist=eval_(args_.e2_dist), 
            fix_causality=False, 
            seed=args_.random_seed
        )
    }

def make_testdata(csv_file, gen_data_params):
    """Create CSV file including artificial data. 
    """
    data = gen_artificial_data(gen_data_params)
    xs = data['xs'].astype('S20')

    if data['causality_true'] == [1, 2]:
        header = ['x1_src', 'x2_dst']
    else:
        header = ['x1_dst', 'x2_src']

    csv_data = np.vstack((header, xs))
    np.savetxt(csv_file, csv_data, fmt='%s', delimiter=',')
    print('Made artificial data and saved as %s.' % csv_file)
