# -*- coding: utf-8 -*-

from .extract_const import extract_const
from .standardize_samples import standardize_samples
from .usr_distrib import usr_distrib, dist_names
from .gendata import GenDataParams, gen_artificial_data

def eval_(arg):
    try:
        return eval(arg)
    except NameError:
        return arg

def wrap_list(arg):
    return arg if type(arg) is list else [arg]
