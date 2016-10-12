# -*- coding: utf-8 -*-

"""Standardize samples.
"""
# Author: Taku Yoshioka, Shohei Shimizu
# License: MIT

from copy import deepcopy
import numpy as np

def standardize_samples(xs, standardize):
    xs = deepcopy(xs)
    if standardize == 'keepratio':
        s = np.std(xs)
        xs = (xs - np.mean(xs, axis=0)) / s
    elif standardize == 'scaling':
        xs = xs / np.std(xs, axis=0)
    elif standardize == 'commonscaling':
        xs = xs / np.std(xs)
    elif standardize is True:
        xs = (xs - np.mean(xs, axis=0)) / np.std(xs, axis=0)
    elif standardize is False:
        xs = xs
    else:
        raise ValueError("Invalid value of standardize: %s" % standardize)

    return xs
