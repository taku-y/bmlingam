# -*- coding: utf-8 -*-

"""Function for causal inference. 
"""
# Author: Taku Yoshioka
# License: MIT

import json

from bmlingam import define_hparam_searchspace, InferParams, \
                     find_best_model, show_hparams

def infer_causality(xs, infer_params, varnames=None, verbose=1):
    """Infer causality based on samples given pair of columns in data.

    :param xs: Sample vectors.
    :type xs: numpy.ndarray, shape=(n_samples, 2)
    :param infer_params: Inference parameters.
    :type infer_params: bmlingam.InferParams
    :param varnames: List of variable names.
    :type varnames: list of str, len(varnames)=2
    :param int verbose: Verbose level.

    The return value is a dicrionary containing the following values:

    .. code:: python

        {
            # Variable names given as input arguments
            # str
            'x1_name': x1_name, 
            'x2_name': x2_name,

            # Sample vectors 
            # numpy.ndarray, shape=(n_samples, 2)
            'xs': xs,

            # Inferred causality
            # [1, 2] (list of int): (var1 -> var2)
            # [2, 1] (list of int): (var2 -> var1)
            'causality': causality, 

            # Inferred causality as a string
            # str
            'causality_str': ('%s -> %s' % (src, dst)),

            # Posterior probability of the selected model
            # float
            'post_prob': post_prob,

            # Hyperparameters of the selected model
            'hparams': hparams_best,

            # Posterior probability of the reverse optimal model
            # float
            'post_prob_rev': post_prob_rev,

            # Hyperparameters of the reverse optimal model
            'hparams_rev': hparams_rev
        }

    """
    assert(type(infer_params) == InferParams)

    if varnames is None:
        varnames = ['var1', 'var2']

    hparamss = define_hparam_searchspace(infer_params)
    sampling_mode = infer_params.sampling_mode
    hparams_best, post_prob, ll, hparams_rev, post_prob_rev, ll_rev = \
        find_best_model(xs, hparamss, sampling_mode)
    causality = hparams_best['causality']

    x1_name = varnames[0]
    x2_name = varnames[1]
    if causality == [1, 2]:
        src, dst = x1_name, x2_name
    else:
        src, dst = x2_name, x1_name

    result = {
        'Infered causality': '{} -> {}'.format(src, dst), 
        '2 * log(p(M)) - log(p(M_rev))': '{}'.format(2 * (ll - ll_rev))
    }

    if 1 <= verbose:
        print(json.dumps(result, indent=2, sort_keys=True))

    if 2 <= verbose:
        print('---- Inference for variables "%s" and "%s" ----' % 
            (x1_name, x2_name))
        print(
            'Inferred  : %s -> %s (posterior prob: %1.3f, loglikelihood: %1.3f)' % 
            (src, dst, post_prob, ll))
        print(
            '(best_rev): %s -> %s (posterior prob: %1.3f, loglikelihood: %1.3f)' % 
            (dst, src, post_prob_rev, ll_rev))
        print('')
        print('Hyper parameters of the optimal model:')
        show_hparams(hparams_best)
        print('')
        print('Hyper parameters of the reverse optimal model:')
        show_hparams(hparams_rev)
        print('')

    return {
        'x1_name': x1_name, 
        'x2_name': x2_name, 
        'xs': xs, 
        'causality': causality, 
        'causality_str': ('%s -> %s' % (src, dst)), 
        'post_prob': post_prob, 
        'hparams': hparams_best, 
        'post_prob_rev': post_prob_rev, 
        'hparams_rev': hparams_rev
    }
