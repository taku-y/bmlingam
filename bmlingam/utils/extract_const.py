# -*- coding: utf-8 -*-

# Author: Taku Yoshioka
# License: MIT

# import re

def extract_const(sample_coef, var_name):
    """Extract float constants from string. 
    """
    if sample_coef.startswith(var_name + '='):
        return float(sample_coef[len(var_name + '='):])
    else:
        raise ValueError(("Invalid variable name: %s\n" % var_name))

    # r = re.compile('%s=(-|)([1-9]\d*|0)(\.\d+)?$' % var_name)
    # m = r.search(sample_coeff)
    # if m is not None:
    #     return float(''.join(m.groups()))
    # else:
    #     return None
