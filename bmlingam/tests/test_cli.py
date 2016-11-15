# -*- coding: utf-8 -*-

# Author: Taku Yoshioka
# License: MIT

import os
import tempfile

from bmlingam.commands.bmlingam_make_testdata \
    import make_testdata, parse_args_bmlingam_make_testdata
from bmlingam.commands.bmlingam_causality \
    import parse_args_bmlingam_causality, bmlingam_causality
from bmlingam.commands.bmlingam_coeff \
    import parse_args_bmlingam_coeff, bmlingam_coeff

def _run_bmlingam_make_testdata(data_file):
    params = parse_args_bmlingam_make_testdata(
        ['--csv_file', data_file]
    )
    make_testdata(**params)

def _run_bmlingam_causality(data_file, model_file):
    params = parse_args_bmlingam_causality(
        [data_file, '--result_dir', '', '--out_optmodelfile', 
         '--optmodel_files', ["{}".format(model_file)], 
         '--n_mc_samples', '100']
    )
    bmlingam_causality(**params)

def _run_bmlingam_coeff(model_file, plot):
    args = [model_file, '--no_save_figure']

    if plot:
        args += ['--plot_figure']
    else:
        args += ['--no_plot_figure']

    params = parse_args_bmlingam_coeff(args)
    bmlingam_coeff(**params)

def test_cli(plot=False):
    try:
        _, data_file = tempfile.mkstemp()
        _, model_file = tempfile.mkstemp()
        print('Temporary files for test_cli()')
        print('data_file: {}'.format(data_file))
        print('model_file: {}\n'.format(model_file))

        _run_bmlingam_make_testdata(data_file)
        _run_bmlingam_causality(data_file, model_file)
        _run_bmlingam_coeff(model_file, plot)

    finally:
        os.remove(data_file)
        os.remove(model_file)
