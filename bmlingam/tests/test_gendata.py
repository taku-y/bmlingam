# -*- coding: utf-8 -*-

# Author: Taku Yoshioka
# License: MIT

import numpy as np
from nose.tools import *
from pprint import pprint
import matplotlib.pyplot as plt

from bmlingam.utils.gendata import gen_artificial_data, GenDataParams

def test_gen_artificial_data(plot=False, n_confounders=1):
    """Only check the function runs without error.
    """
    gen_data_params = GenDataParams(
        e1_dist=['uniform'], 
        f1_coef=['r2intervals' for _ in range(n_confounders)], 
        f2_coef=['r2intervals' for _ in range(n_confounders)], 
        conf_dist=[['all'] for _ in range(n_confounders)]
    )
    n_samples = gen_data_params.n_samples
    data = gen_artificial_data(gen_data_params)

    xs = data['xs']
    confs = data['confs']

    assert(data['xs'].shape == (n_samples, 2))

    print('xs.shape   = {}'.format(xs.shape))
    print('std(x1)    = {}'.format(np.std(xs[:, 0])))
    print('std(x2)    = {}'.format(np.std(xs[:, 1])))   
    print('std(conf1) = {}'.format(np.std(confs[:, 0])))
    print('std(conf2) = {}'.format(np.std(confs[:, 1])))

    print('')
    pprint(vars(gen_data_params))

    plt.figure()
    plt.scatter(xs[:, 0], xs[:, 1])
    plt.xlabel('x1')
    plt.xlabel('x2')
    plt.title('Observations')

    plt.figure()
    plt.hist(data['es'])
    plt.title('Errors')

    plt.figure()
    plt.hist(data['confs'])
    plt.title('Confounders')
