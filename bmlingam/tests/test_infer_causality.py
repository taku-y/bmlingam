# -*- coding: utf-8 -*-

# Author: Taku Yoshioka
# License: MIT

import numpy as np
import os
import tempfile

from bmlingam.commands.bmlingam_causality import infer_causality, \
                                                 parse_args_bmlingam_causality, \
                                                 load_data
from bmlingam.utils.gendata import gen_artificial_data, GenDataParams

def _gen_artificial_data_csv(csv_file, gen_data_params):
    data = gen_artificial_data(gen_data_params)
    xs = data['xs']

    if data['causality_true'] == [1, 2]:
        header = 'x1_src,x2_dst'
    else:
        header = 'x1_dst,x2_src'

    np.savetxt(csv_file, xs, fmt='%s', delimiter=',', header=header, 
               comments='')

    print('Artificial data is generated and saved as %s' % csv_file)

def test_infer_causality():
    try:
        # Generate artificial data and save as csv file
        _, csv_file = tempfile.mkstemp()
        _gen_artificial_data_csv(csv_file, gen_data_params=GenDataParams())

        # Parse args
        params_ = parse_args_bmlingam_causality(
            [csv_file, '--result_dir', '', '--no_out_optmodelfile', 
             '--sampling_mode', 'cache_mp4']
        )

        # Load data
        data = load_data(csv_file, params_['col_names']) # Pandas dataframe
        xs = np.array(data.ix[:, :2])

        # Do inference
        params = {
            'xs': xs,
            'infer_params': params_['infer_params'],
            'varnames': data.columns.values[:2],
            'verbose': 1
        }
        infer_causality(**params)

    finally:
        os.remove(csv_file)
