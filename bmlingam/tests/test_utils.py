# -*- coding: utf-8 -*-

# Author: Taku Yoshioka
# License: MIT

from nose.tools import *

from bmlingam.utils import extract_const

def test_extract_const():
    var1 = extract_const('var1=10.0', 'var1')
    assert(var1 == 10.0)

@raises(ValueError)
def test_extract_const2():
    extract_const('var2=12.0', 'var1')
