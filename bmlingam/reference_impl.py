# -*- coding: utf-8 -*-

"""Perform analysis using bayesian mixed LiNGAM model.
"""
# Author: Taku Yoshioka
# License: MIT

import numpy as np
from scipy.misc import logsumexp

def bayesmixedlingam_ref(
    xs, n_theta_sampling, P_M1, P_M2, prior_indvdl, rng):
    """A reference implementation of Bayesian mixed-LiNGAM analysis. 

    The following two models are compared: 

    - M1: Model1 x1->x2
    - M2: Model2 x1<-x2

    Return values are obtained as follows: 

    .. code:: python
        
        logP_M_givenD = [np.max(logPs_M1_givenD), np.max(logPs_M2_givenD)]
        if logP_M_givenD[1] < logP_M_givenD[0]:
            kest = [1, 2] # M1: Model1 x1->x2
        else:
            kest = [2, 1] # M2: Model2 x1<-x2

    In the above code, :code:`np.max()` takes the maximum of the 
    marginal log-likelihoods computed over the search space of 
    various configurations of hyperparameters. 

    :code:`prior_indvdl` is :code:`t` or :code:`gauss`. 

    :param xs: Observed samples. 
    :type xs: ndarray, shape=(n_samples, 2), dtype=float
    :param int n_theta_sampling: Number of sampling iterations.
    :param float P_M1: Prior probability selecting model M1. 
    :param float P_M2: Prior probability selecting model M2. 
    :param str prior_indvdl: Prior type of individual-specific effects.
    :param numpy.random.RandomState: Random number generator.
    :return: (kest, logP_M_givenD)

    """        
    # ------ Define search range over hyperparameters ------
    max_c = 10
    c = np.std(xs, axis=0)
    cs = np.arange(0, 1, .2)
    v_indvdls_1 = (c[0] * cs)**2
    v_indvdls_2 = (c[1] * cs)**2

    # Covariance matrices
    L_covs = []
    for L_cov_21 in [-.9, -.7, -.5, -.3, 0, .3, .5, .7, .9]:
        L_cov = np.eye(2)
        L_cov[0, 1] = L_cov_21
        L_cov[1, 0] = L_cov_21
        L_covs.append(L_cov)
    del L_cov

    # ------ Compute marginal lls over search range ------
    logPs = [comp_logP_M12_ref(
        xs, n_theta_sampling, P_M1, P_M2, v_indvdl_1, v_indvdl_2, L_cov, 
        max_c, prior_indvdl, rng)
        for v_indvdl_1 in v_indvdls_1
        for v_indvdl_2 in v_indvdls_2
        for L_cov in L_covs]
    logPs_M1_givenD = np.array([logP[0] for logP in logPs])
    logPs_M2_givenD = np.array([logP[1] for logP in logPs])

    # ------ Compare models ------
    logP_M_givenD = [np.max(logPs_M1_givenD), np.max(logPs_M2_givenD)]
    if logP_M_givenD[1] < logP_M_givenD[0]:
        kest = [1, 2] # M1: Model1 x1->x2
    else:
        kest = [2, 1] # M2: Model2 x1<-x2

    return kest, logP_M_givenD

def comp_logP_M12_ref(
    xs, n_theta_sampling, P_M1, P_M2, v_indvdl_1, v_indvdl_2, L_cov, max_c, 
    prior_indvdl, rng):
    """Compute :math:`\\log p({\\cal M}_{r}|{\\cal D}) (r=1, 2)` 
    for given two models (up to constant :math:`\\log p({\\cal D})`). 

    Integration over :math:`\\theta` is performed by naive MonteCalro 
    sampling. 

    Return values logP_M1_givenD and logP_M2_givenD denotes
    :math:`\\log p({\\cal D}|{\\cal M}_{r})`

    :param xs: Observed samples. 
    :type xs: ndarray, shape=(n_samples, 2), dtype=float
    :param int n_theta_sampling: Number of sampling iterations.
    :param float P_M1: Prior probability selecting model M1. 
    :param float P_M2: Prior probability selecting model M2. 
    :param v_indvdl_1: Mean of individual-specific effects.
    :param v_indvdl_2: Mean of individual-specific effects.
    :param L_cov: Covariance matrix of individual-specific effects.
    :type L_cov: ndarray, shape=(2, 2), dtype=float
    :param max_c: ??
    :param str prior_indvdl: Prior type of individual-specific effects.
    :param np.random.RandomState rng: Random number generator. 
    :return: logP_M1_givenD, logP_M2_givenD
    :rtype: (float, float)

    """
    # ------ Select hyperparameters ------
    n_samples = xs.shape[0]
    c = np.std(xs, axis=0)
    tau_cmmn = np.array([(c[0] * max_c)**2, (c[1] * max_c)**2])

    # ------ Compute log p(D | M) ------
    df_L = 6
    Sigma = df_L / (df_L - 2) * L_cov
    if prior_indvdl == 't':
        L = multivariatet(0, L_cov, df_L, n_theta_sampling * n_samples, rng)
    elif prior_indvdl == 'gauss':
        L = rng.multivariate_normal(
            np.zeros(L_cov.shape[0]), L_cov, n_theta_sampling * n_samples)
    else:
        raise ValueError('Invalid value of prior_indvdl.')

    # ------ Normalize variables to variance one ------
    L = L * np.diag(Sigma)**(-0.5)
    L_mu1 = L[:, 0].reshape(n_theta_sampling, n_samples)
    L_mu2 = L[:, 1].reshape(n_theta_sampling, n_samples)

    # ------ Generate theta ------
    gen_randnones = lambda: (
        rng.randn(n_theta_sampling, 1) * np.ones((1, n_samples)))
    mu1 = np.sqrt(tau_cmmn[0]) * gen_randnones()
    mu2 = np.sqrt(tau_cmmn[1]) * gen_randnones()
    h1 = np.abs(np.sqrt(tau_cmmn[0]) * gen_randnones())
    h2 = np.abs(np.sqrt(tau_cmmn[1]) * gen_randnones())
    b21 = np.sqrt(tau_cmmn[1]) * gen_randnones()
    b12 = np.sqrt(tau_cmmn[0]) * gen_randnones()

    # ------ Update mu1 and mu2 adding individual-specific effects ------
    mu1 = mu1 + np.sqrt(v_indvdl_1) * L_mu1
    mu2 = mu2 + np.sqrt(v_indvdl_2) * L_mu2

    # ------ Compute probabilities of samples given parameters ------
    exp_nakami_xis_givenMgammatheta1 = \
        comp_exp_nakami_xis_givenMgammatheta(
            1, n_theta_sampling, xs, mu1, mu2, b21, None, h1, h2)
    exp_nakami_xis_givenMgammatheta2 = \
        comp_exp_nakami_xis_givenMgammatheta(
            2, n_theta_sampling, xs, mu1, mu2, None, b12, h1, h2 )

    # ------ Averaging over MonteCarlo samples ------
    logP_D_givenM1 = np.sum(
        logsumexp(exp_nakami_xis_givenMgammatheta1, 0) - 
        np.log(n_theta_sampling))
    logP_D_givenM2 = np.sum(
        logsumexp(exp_nakami_xis_givenMgammatheta2, 0) - 
        np.log(n_theta_sampling))
    
    # ------ Log marginal probability (up to const) ------
    logP_M1_givenD = logP_D_givenM1 + np.log(P_M1)
    logP_M2_givenD = logP_D_givenM2 + np.log(P_M2)

    return logP_M1_givenD, logP_M2_givenD

def comp_exp_nakami_xis_givenMgammatheta(
    M_id, n_theta_sampling, xs, mu1, mu2, b21, b12, h1, h2):
    """Compute probabilities of samples given parameters: 
    :math:`p(x_{l} - \\mu_{l} - b_{lm}x_{m}|{\\bf\\theta})`. 

    It is assumed a Laplace distribution as noise model. 

    :param int M_id: Model ID (1 or 2).
    :param int n_theta_sampling: Number of sampling iterations.
    :param xs: Observed samples. 
    :type xs: ndarray, shape=(n_samples, 2), dtype=float
    :param float mu1: Mean of the 1st signal.
    :param float mu2: Mean of the 2nd signal.
    :param float b21: Connection strength: 1 -> 2.
    :param float b12: Connection strength: 2 -> 1.
    :param float h1: Std of the 1st signal.
    :param float h2: Std of the 2nd signal.
    :return: Probabilities of samples given parameters. 
    :rtype: ndarray, shape=(n_theta_sampling, n_samples), dtype=float

    """
    gen_ones = lambda: np.ones((n_theta_sampling, 1))

    if M_id == 1:
        exp_nakami_xis_givenMgammatheta = (
            logP((gen_ones() * xs[:, 0].T - mu1) / h1) +
            logP((gen_ones() * xs[:, 1].T - mu2 - 
                  b21 * (gen_ones() * xs[:, 0].T - mu1)) / h2))
    elif M_id == 2:
        exp_nakami_xis_givenMgammatheta = (
            logP((gen_ones() * xs[:, 0].T - mu1 -
                  b12 * (gen_ones() * xs[:, 1].T - mu2)) / h1) +
            logP((gen_ones() * xs[:, 1].T - mu2) / h2))
    else:
        raise ValueError('Invalid value of M_id: %s.' + str(M_id))

    exp_nakami_xis_givenMgammatheta = \
        exp_nakami_xis_givenMgammatheta - np.log(h1) - np.log(h2)

    return exp_nakami_xis_givenMgammatheta

def logP(e):
    """Return pdf of Laplace distribution with mean 0 and variance 1. 

    :param e: Sample values.
    :type e: ndarray, shape=(n_theta_sampling, n_samples)
    :return: pdf values at given samples. 
    :rtype: ndarray, shape=(n_theta_sampling, n_samples)

    """
    b = 1 / np.sqrt(2)
    out = (-np.log(2 * b) * np.ones(e.shape) - np.abs(e) / b)
    
    return out

def multivariatet(mu, Sigma, N, M, rng):
    """Return a sample (or samples) from the multivariate t distribution.

    This function is adopted from 
    http://kennychowdhary.me/2013/03/python-code-to-generate-samples-from-multivariate-t/.

    :param mu: Mean.
    :type mu: ndarray, shape=(n_dim,), dtype=float
    :param Sigma: Scaling matrix.
    :type Sigma: ndarray, shape=(n_dim, n_dim), dtype=float
    :param float N: Degrees of freedom.
    :param int M: Number of samples to produce. 
    :param np.random.RandomState rng: Random number generator. 
    :return: M samples of (n_dim)-dimensional multivariate t distribution.
    :rtype: ndarray, shape=(n_samples, n_dim), dtype=float

    """
    d = len(Sigma)
    g = np.tile(np.random.gamma(N/2., 2./N, M), (d, 1)).T
    Z = rng.multivariate_normal(np.zeros(d), Sigma, M)
    return mu + Z / np.sqrt(g)

