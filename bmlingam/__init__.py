# -*- coding: utf-8 -*-

"""The :mod:`lingam` module includes a Bayesian mixed-LiNGAM model.
"""
# Author: Taku Yoshioka, Shohei Shimizu
# License: MIT

import matplotlib
import os
import pandas as pd
import re

from bmlingam.find_best_model import find_best_model
from bmlingam.bmlingam_pm3 import do_mcmc_bmlingam, MCMCParams

from bmlingam.bmlingam_np import comp_logP_bmlingam_np

from bmlingam.hparam import define_hparam_searchspace, show_hparams, InferParams
from bmlingam.bmlingam_np import comp_logP_M12_np
from bmlingam.bmlingam_pm3 import get_pm3_model_bmlingam
from bmlingam.infer_causality import infer_causality
from bmlingam.infer_coef_posterior import infer_coef_posterior

# Set backend when there is no display
display = os.environ.get('DISPLAY')
if display is None or not re.search(':\d', display):
    matplotlib.use('Agg')

"""Utility functions
"""

def save_pklz(fileName, obj):
    """Save serializable (able to be pickled) Python object into the file. 

    :param str filename: Filename. 
    :param obj: Python object. 
    """
    import pickle
    import gzip

    f = gzip.GzipFile(fileName, 'wb')
    f.write(pickle.dumps(obj))
    f.close()

    return

def load_pklz(fileName):
    """Load Python object from the file created by save_pklz(). 

    :param str filename: Filename. 
    :return: Python object saved in the file. 
    """
    import pickle
    import gzip

    f   = gzip.GzipFile(fileName, 'rb')
    obj = pickle.load(f)
    f.close()

    return obj

def load_data(csv_file, col_names):
    """Load data from csv file.
    """
    if col_names is None:
        data = pd.read_csv(csv_file, sep=None, engine='python')
    elif (col_names == 'auto') or (col_names[0] == 'auto'):
        data = pd.read_csv(csv_file, header=None, sep=None, engine='python')
        n_cols = data.columns # Num of columns in the dataframe
        data.columns = ['x%d' % i for i in range(n_cols)]
    else:
        data = pd.read_csv(csv_file, header=None, sep=None, engine='python')
        data.columns = col_names

    print('---- Data ----')
    print('Data loaded from %s.' % csv_file)
    print('Data contains %d samples.' % len(data.index))
    print('Variable names: %s' % str(data.columns.values))
    print('')

    return data

