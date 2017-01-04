#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

classifiers = """\
Development Status :: 3 - Alpha
Intended Audience :: Education
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Programming Language :: Python
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
Programming Language :: Python :: 2.7
"""

setup(
    name = 'BMLiNGAM',
    version = '0.1.5',
    description = 'Software for causal estimation.',
    long_description = '',
    classifiers = classifiers,
    author = 'Taku Yoshioka, Shohei Shimizu',
    author_email = 'taku.yoshioka.4096@gmail.com',
    url = '',
    license = 'MIT',
    platforms = ['Windows', 'Linux', 'Solaris', 'Mac OS-X', 'Unix'],
    packages = ['bmlingam', 'bmlingam.commands', 'bmlingam.tests', 
                'bmlingam.utils'], 
    install_requires = ['numpy>=1.6.2', 'scipy>=0.11', 'matplotlib>=1.5.3', 
                        'pandas>=0.18', 'parse', 'pymc3>=3.0rc2'],
    scripts = ['bin/bmlingam-causality', 'bin/bmlingam-coeff', 
               'bin/bmlingam-make-testdata'],
)
