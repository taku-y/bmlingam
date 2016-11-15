# -*- coding: utf-8 -*-

"""Define function to make hyperparameter search space.
"""
# Author: Taku Yoshioka
# License: MIT

import numpy as np

"""Define search space of hyperparameters
"""

class InferParams(object):
    """Parameters for causal inference with the bayesian mixed LiNGAM.

    The fields of this class are grouped into two sets. The one is the set
    of parameters that define the candidates of models, i.e., possible value
    of hyperparameters. 

    :param prior_indvdls: Distributions of individual specific effects.
        Possible values are :code:`['t']`, :code:`['gauss']`, :code:`['gg']`, 
        :code:`['t', 'gauss']`, or :code:`['t', 'gg']`. The default value is
        :code:`['t']`. :code:`gg` means generalized Gaussian distribution.
    :type prior_indvdls: list of string
    :param cs: Scales of the standard deviations of the individual specific 
        effects. The default value is :code:`[0, 0.2, 0.4, 0.6, 0.8, 1.0]`.
    :type cs: list of float
    :param L_cov_21s: Correlation coefficients of individual specific effects. 
        The default value is :code:`[-.9, -.7, -.5, -.3, 0, .3, .5, .7, .9]`.
    :type L_cov_21s: list of float
    :param betas_indvdl: Beta values of generalized Gaussian distributions
        for individual specific effects. The default value is
        :code:`[0.25, 0.5, 0.75, 1.]`.
    :param betas_noise: Beta values of generalized Gaussian distributions
        for noise. The default value is :code:`[0.25, 0.5, 0.75, 1.]`.
    :param causalities: Causal directions to be tested.
        The default value is :code:`x1->x2, x2->x1`.

    The other includes parameters applied to all candidate models.

    :param int seed: Seed of random number generator. The default value is 0.
    :param bool standardize: If true (default), samples are standardized. 
    :param bool subtract_mu_reg: Specify regression variable.
        The default value is :code:`False`.
    :param bool fix_mu_zero: If :code:`True` (default), the common interception
        is fixed to 0.
    :param prior_var_mu: Set prior variance of common interceptions.
        The default value is :code:`auto`.
    :type prior_var_mu: 'auto' (str literal) or float
    :param float max_c: Scale constant on tau_cmmn.
        The default value is :code:`1.0`.
    :param int n_mc_samples: Number of Monte Carlo samples.
        The default value is :code:`10000`.
    :param float P_M1: Prior probability selecting model M1.
        The default value is :code:`0.5`.
    :param float P_M2: Prior probability selecting model M2. 
        The default value is :code:`0.5`.
    :param dist_noise: Noise distribution. The default vaule is :code:`gg`.
    :type dist_noise: str, :code:`'laplace'` or :code:`'gg'`. 
    :param float df_indvdl: Degrees of freedom of T distibution. 
    :param prior_scale': Prior distribution of scale variable.
    :type prior_scale: str, :code:`'tr_normal'` or :code:`'log_normal'`
        The default value is :code:`log_normal`.
    :param float scale_coeff': Scale of prior std of regression coefficient.
        The default value is :code:`1.0`.
    :param str sampling_mode: Sampling mode for MC integration.
        The default value is :code:`normal`, in which random numbers are 
        generated for each candidate models. If :code:`cache[_mp(2, 4, 8)]`,
        random numbers are cached and reused for evaluation of each candidate
        models. This mode is faster than :code:`normal`. :code:`_mp(2, 4, 8)`
        means using multiprocesses (2, 4 or 8 processes) for evaluation of 
        candidate models in parallel.
    """
    def __init__(
        self, 
        seed=0, 
        standardize=True, 
        subtract_mu_reg=False, 
        fix_mu_zero=True, 
        prior_var_mu='auto', 
        max_c=1., 
        n_mc_samples=10000, 
        P_M1=0.5, 
        P_M2=0.5, 
        dist_noise='gg', 
        df_indvdl=8., 
        prior_scale='log_normal', 
        scale_coeff=1.0, 
        prior_indvdls=['t'], 
        cs=np.arange(0, 1.2, .2), 
        L_cov_21s=[-.9, -.7, -.5, -.3, 0, .3, .5, .7, .9], 
        betas_indvdl=[0.25, 0.5, 0.75, 1.], 
        betas_noise=[0.25, 0.5, 0.75, 1.], 
        causalities='x1->x2, x2->x1', 
        sampling_mode='normal'):

        if causalities == 'x1->x2':
            causalities = [[1, 2]]
        elif causalities == 'x2->x1':
            causalities = [[2, 1]]
        elif causalities == 'x1->x2, x2->x1':
            causalities = [[1, 2], [2, 1]]
        elif type(causalities) is str:
            raise ValueError("Invalid value of causalities: %s" % causalities)

        self._seed = seed
        self._standardize = standardize
        self._subtract_mu_reg = subtract_mu_reg
        self._fix_mu_zero = fix_mu_zero
        self._prior_var_mu = prior_var_mu
        self._max_c = max_c
        self._n_mc_samples = n_mc_samples 
        self._P_M1 = P_M1 
        self._P_M2 = P_M2 
        self._dist_noise = dist_noise 
        self._df_indvdl = df_indvdl
        self._prior_scale = prior_scale
        self._scale_coeff = scale_coeff
        
        self._prior_indvdls = prior_indvdls
        self._cs = cs
        self._L_cov_21s = L_cov_21s
        self._betas_indvdl = betas_indvdl
        self._betas_noise = betas_noise
        self._causalities = causalities

        self._sampling_mode = sampling_mode

    @property
    def seed(self):
        return self._seed
    
    @property
    def standardize(self):
        return self._standardize

    @standardize.setter
    def standardize(self, value):
        self._standardize = value

    @property
    def subtract_mu_reg(self):
        return self._subtract_mu_reg
    
    @property
    def fix_mu_zero(self):
        return self._fix_mu_zero

    @fix_mu_zero.setter
    def fix_mu_zero(self, value):
        self._fix_mu_zero = value

    @property
    def prior_var_mu(self):
        return self._prior_var_mu
    
    @property
    def max_c(self):
        return self._max_c

    @max_c.setter
    def max_c(self, value):
        self._max_c = value

    @property
    def n_mc_samples(self):
        return self._n_mc_samples

    @n_mc_samples.setter
    def n_mc_samples(self, value):
        self._n_mc_samples = value

    @property
    def P_M1(self):
        return self._P_M1

    @property
    def P_M2(self):
        return self._P_M2

    @property
    def df_indvdl(self):
        return self._df_indvdl

    @df_indvdl.setter
    def df_indvdl(self, value):
        self._df_indvdl = value
    
    @property
    def prior_scale(self):
        return self._prior_scale
    
    @prior_scale.setter
    def prior_scale(self, value):
        self._prior_scale = value
    
    @property
    def prior_indvdls(self):
        return self._prior_indvdls

    @property
    def scale_coeff(self):
        return self._scale_coeff

    @scale_coeff.setter
    def scale_coeff(self, value):
        self._scale_coeff = value
    
    @property
    def cs(self):
        return self._cs

    @cs.setter
    def cs(self, value):
        self._cs = value

    @property
    def L_cov_21s(self):
        return self._L_cov_21s

    @L_cov_21s.setter
    def L_cov_21s(self, value):
        self._L_cov_21s = value

    @property
    def dist_noise(self):
        return self._dist_noise

    @dist_noise.setter
    def dist_noise(self, value):
        self._dist_noise = value

    @property
    def betas_indvdl(self):
        return self._betas_indvdl

    @property
    def betas_noise(self):
        return self._betas_noise

    @property
    def causalities(self):
        return self._causalities

    @property
    def sampling_mode(self):
        return self._sampling_mode

    @sampling_mode.setter
    def sampling_mode(self, value):
        self._sampling_mode = value

def define_hparam_searchspace(infer_params):
    """Returns a list of hyperparameter sets. 

    Search space is defined by the following fields in :code:`infer_params`:

    - :code:`prior_indvdls`
    - :code:`cs`
    - :code:`L_cov_21s`
    - :code:`betas_indvdl`
    - :code:`betas_noise`

    Other fields are treated as constants.  

    All of the returned values, i.e. hyperparameters, are used to calculate 
    log marginal probabilities and the maximal value is taken. 

    :param InferParams infer_params: Parameters defining hyperparameter set.

    Returned value is a list of dicts, each of that is a set of 
    hyperparameters. It is created by the following code:

    .. code-block:: python

        hparamss = [
            {
                # Fixed hyperparameters
                'seed': seed, 
                'standardize': standardize, 
                'subtract_mu_reg': subtract_mu_reg, 
                'fix_mu_zero': fix_mu_zero, 
                'prior_var_mu': prior_var_mu, 
                'max_c': max_c, 
                'n_mc_samples': n_mc_samples, 
                'P_M1': P_M1, 
                'P_M2': P_M2, 
                'dist_noise': dist_noise, 
                'df_indvdl': df_indvdl, 
                'prior_scale': prior_scale, 

                # Varied hyperparameters
                'causality': causality, 
                'prior_indvdl': prior_indvdl, 
                'v_indvdl_1': v_indvdl_1, 
                'v_indvdl_2': v_indvdl_2, 
                'L_cov': L_cov, 
                'beta_coeff': beta_coeff, 
                'beta_noise': beta_noise
            }
            for prior_indvdl in prior_indvdls
            for v_indvdl_1 in cs
            for v_indvdl_2 in cs
            for L_cov in L_covs
            for causality in causalities
            for beta_coeff in (betas_indvdl if prior_indvdl == 'gg' else [None])
            for beta_noise in (betas_noise if dist_noise == 'gg' else [None])]

    :return: List of hyperparameter sets. 
    :rtype: list of dicts
    """
    assert(type(infer_params) == InferParams)
    return define_hparam_searchspace_main(
        seed=infer_params.seed, 
        standardize=infer_params.standardize, 
        subtract_mu_reg=infer_params.subtract_mu_reg, 
        fix_mu_zero=infer_params.fix_mu_zero, 
        prior_var_mu=infer_params.prior_var_mu, 
        max_c=infer_params.max_c, 
        n_mc_samples=infer_params.n_mc_samples, 
        P_M1=infer_params.P_M1, 
        P_M2=infer_params.P_M2, 
        df_indvdl=infer_params.df_indvdl, 
        dist_noise=infer_params.dist_noise, 
        prior_scale=infer_params.prior_scale, 
        prior_indvdls=infer_params.prior_indvdls, 
        cs=infer_params.cs, 
        L_cov_21s=infer_params.L_cov_21s, 
        betas_indvdl=infer_params.betas_indvdl, 
        betas_noise=infer_params.betas_noise, 
        causalities=infer_params.causalities, 
        scale_coeff=infer_params.scale_coeff
    )

def define_hparam_searchspace_main(
    seed=0, standardize=True, subtract_mu_reg=True, fix_mu_zero=False, 
    prior_var_mu='auto', max_c=10, n_mc_samples=1000, P_M1=0.5, P_M2=0.5, 
    prior_indvdls=['t'], cs=np.arange(0, 1, .2), 
    L_cov_21s=[-.9, -.7, -.5, -.3, 0, .3, .5, .7, .9], 
    tied_sampling=False, dist_noise='laplace', 
    betas_indvdl=[0.25, 0.5, 0.75, 1.], betas_noise=[0.25, 0.5, 0.75, 1.], 
    df_indvdl=8., prior_scale='tr_normal', causalities=[[1, 2], [2, 1]], 
    scale_coeff=1.0):
    u"""Return set of hyperparameters. 

    Search space is defined by input arguments :code:`prior_indvdls`, 
    :code:`cs`, :code:`L_cov_21s`, and :code:`dist_noise`. For generalized 
    Gaussian distribution, :code:`betas_indvdl` and :code:`betas_noise` are 
    also involved to be searched. Others are treated as constants.  

    All of the returned values, i.e. hyperparameters, are used to calculate 
    log marginal probabilities and the maximal value is taken. 
    """
    if (P_M1 + P_M2) != 1.:
        raise ValueError("(P_M1 + P_M2) should be 1.")

    # Individual specific effects
    v_indvdls_1 = cs
    v_indvdls_2 = cs

    # Hyperparameter search space
    if tied_sampling:
        raise NotImplementedError('tied_sampling is no longer be supported.')

    hparamss = [
        {
            # Fixed hyperparameters
            'seed': seed, 
            'standardize': standardize, 
            'subtract_mu_reg': subtract_mu_reg, 
            'fix_mu_zero': fix_mu_zero, 
            'prior_var_mu': prior_var_mu, 
            'max_c': max_c, 
            'n_mc_samples': n_mc_samples, 
            'P_M1': P_M1, 
            'P_M2': P_M2, 
            'dist_noise': dist_noise, 
            'df_indvdl': df_indvdl, 
            'prior_scale': prior_scale, 
            'scale_coeff': scale_coeff, 

            # Varied hyperparameters
            'causality': causality, 
            'prior_indvdl': prior_indvdl, 
            'v_indvdl_1': v_indvdl_1, 
            'v_indvdl_2': v_indvdl_2, 
            'L_cov_21': L_cov_21, 
            'beta_coeff': beta_coeff, 
            'beta_noise': beta_noise
        }
    for prior_indvdl in prior_indvdls
    for v_indvdl_1 in v_indvdls_1 
    for v_indvdl_2 in v_indvdls_2 
    for L_cov_21 in L_cov_21s
    for causality in causalities
    for beta_coeff in (betas_indvdl if prior_indvdl == 'gg' else [None])
    for beta_noise in (betas_noise if dist_noise == 'gg' else [None])]

    # ---- Prior distribution ----
    n_models = len(hparamss)
    for hparams in hparamss:
        hparams.update({
            'P_M1': P_M1 * (2. / n_models), 
            'P_M2': P_M2 * (2. / n_models)
        })

    return hparamss

"""Supplemental function
"""

def show_hparams(hparams):
    """Show hyper-parameter values. 
    """
    prior_indvdl = hparams['prior_indvdl']

    if hparams['prior_indvdl'] == 'gauss':
        _show_hparams_gauss(hparams)
    elif hparams['prior_indvdl'] == 't':
        _show_hparams_t(hparams)
    elif hparams['prior_indvdl'] == 'gg':
        _show_hparams_gg(hparams)        
    else:
        raise ValueError("Invalid value of prior_indvdl: %s" % prior_indvdl)
    _show_hparam_noise(hparams)

"""Prior_indvdl == 'gauss'
"""

def _show_hparams_gauss(hparams):
    def _str_causality(c):
        if c == [1, 2]:
            return 'var1 -> var2'
        else:
            return 'var2 -> var1'

    print('Causality      : %s' % _str_causality(hparams['causality']))
    print('Standardize    : %s' % str(hparams['standardize']))
    print('subtract_mu_reg: %s' % str(hparams['subtract_mu_reg']))
    print('fix_mu_zero    : %s' % str(hparams['fix_mu_zero']))
    print('prior_var_mu   : %s' % str(hparams['prior_var_mu']))
    print('prior_indvdl   : %s' % hparams['prior_indvdl'])
    print('v_indvdl_1     : %f' % hparams['v_indvdl_1'])
    print('v_indvdl_2     : %f' % hparams['v_indvdl_2'])
    print('df_indvdl      : %f' % hparams['df_indvdl'])
    print('L_cov12/21     : %f' % hparams['L_cov_21'])

"""Prior_indvdl == 't'
"""

def _show_hparams_t(hparams):
    _show_hparams_gauss(hparams) # Show same values

"""Prior_indvdl == 'gg'
"""

def _show_hparams_gg(hparams):
    def _str_causality(c):
        if c == [1, 2]:
            return 'var1 -> var2'
        else:
            return 'var2 -> var1'

    print('Causality      : %s' % _str_causality(hparams['causality']))
    print('Standardize    : %s' % str(hparams['standardize']))
    print('subtract_mu_reg: %s' % str(hparams['subtract_mu_reg']))
    print('fix_mu_zero    : %s' % str(hparams['fix_mu_zero']))
    print('prior_indvdl   : %s' % hparams['prior_indvdl'])
    print('prior_var_mu   : %s' % str(hparams['prior_var_mu']))
    print('beta_coeff     : %s' % hparams['beta_coeff'])
    print('v_indvdl_1     : %f' % hparams['v_indvdl_1'])
    print('v_indvdl_2     : %f' % hparams['v_indvdl_2'])
    print('df_indvdl      : %f' % hparams['df_indvdl'])
    print('L_cov12/21     : %f' % hparams['L_cov'][0, 1])
    print('beta_noise     : %f' % hparams['beta_noise'])

"""Noise model
"""

def _show_hparam_noise(hparams):
    print('dist_noise     : %s' % hparams['dist_noise'])
    if hparams['dist_noise'] == 'gg':
        print('beta_noise     : %f' % hparams['beta_noise'])
    else:
        pass
