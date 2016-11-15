.. lingam documentation master file, created by
    sphinx-quickstart on Tue Mar 17 09:18:16 2015.
    You can adapt this file completely to your liking, but it should at least
    contain the root `toctree` directive.

BMLiNGAM
===============================================================================
BMLiNGAM is a Python software for causal inference between pairs of variables. The estimation method of this software is based on a Bayesian mixed LiNGAM model, proposed in the paper of Shimizu and Bollen (2014). 

This software is provided as command-line tools. The inference is performed on data given as a CSV file. In addition, the posterior distribution of the regression coefficient, relating two variables in the data, can also be estimated by MCMC. 

This document describes installation, usage of this sofrware, and the interpretation of the results. 

.. toctree:: 
    :maxdepth: 2

    installation
    tutorial
    notebook/expr/20160915/20160822-eval-bml.ipynb
    model
    license
    api
    citation
