# -*- coding: utf-8 -*-

"""PyMC3 probability model of LiNGAM.
"""
# Author: Taku Yoshioka
# License: MIT

import numpy as np
from parse import parse

from bmlingam.prob import GeneralizedGaussian
from bmlingam.utils import standardize_samples

class MCMCParams(object):
    """Parameters for MCMC to estimate regression coefficient.

    :param int n_burn: Samples in burn-in period. 
        The default value is :code:`10000`.
    :param int n_mcmc_samples: Samples in MCMC (after burn-in).
        The default value is :code:`10000`.
    :param int seed_burn: Random seed for burn-in period.
        The default value is :code:`1`.
    :param int seed: Random seed for MCMC.
        The default value is :code:`2`.
    :param int verbose: Verbose level. The default value is 0 (silent).
    """
    def __init__(
        self, 
        n_burn=10000, 
        n_mcmc_samples=10000, 
        seed_burn=1, 
        seed=2, 
        verbose=0, 
        nanguard=False): # verbose == 0 means silent

        self._n_burn = n_burn
        self._n_mcmc_samples = n_mcmc_samples
        self._seed_burn = seed_burn
        self._seed = seed
        self._verbose = verbose
        self._nanguard = nanguard

    @property
    def n_burn(self):
        return self._n_burn

    @n_burn.setter
    def n_burn(self, value):
        self._n_burn = value

    @property
    def n_mcmc_samples(self):
        return self._n_mcmc_samples

    @n_mcmc_samples.setter
    def n_mcmc_samples(self, value):
        self._n_mcmc_samples = value

    @property
    def seed_burn(self):
        return self._seed_burn

    @property
    def seed(self):
        return self._seed

    @property
    def verbose(self):
        return self._verbose

    @property
    def nanguard(self):
        return self._nanguard

def do_mcmc_bmlingam(xs, hparams, mcmc_params):
    """Do MCMC for sampling posterior of bmlingam coefficient.

    Example: 

    .. code:: python

        mcmc_params = MCMCParams(
            n_burn=10000,     # Samples in burn-in period
            n_mcmc_samples=10000, # Samples in MCMC (after burn-in)
            seed_burn=1, # Random seed for burn-in period
            seed=2 # Random seed for MCMC
        ) 
        trace = do_mcmc_bmlingam(data['xs'], hparams, mcmc_params)
        b_post = np.mean(trace['b'])

    :code:`xs` is the numpy.ndarray containing samples. 

    :param xs: Data array. 
    :type xs: numpy.ndarray, shape=(n_samples, 2)

    :code:`hparams` is a dict including hyperparameters. 
    See :func:`bmlingam.hparam.define_hparam_searchspace`. 

    :param hparams: Set of hyperparameters.
    :type hparams: dict

    :code:`mcmc_params` includes parameters for MCMC. 

    :param mcmc_params: Parameters for MCMC. 
    :type mcmc_params: :class:`bmlingam.MCMCParams`
    """
    assert(type(mcmc_params) == MCMCParams)

    # ---- Import PyMC3 modules when required ----
    from pymc3 import Metropolis, sample

    # ---- Standardization ----
    scale_ratio = np.std(xs[:, 1]) / np.std(xs[:, 0])
    xs = standardize_samples(xs, hparams['standardize'])

    model = get_pm3_model_bmlingam(xs, hparams, mcmc_params.verbose)

    # ---- MCMC sampling ----
    with model:
        # Burn-in
        # start = find_MAP()
        step = Metropolis()
        trace = sample(
            mcmc_params.n_burn, step, random_seed=mcmc_params.seed_burn, 
            progressbar=False
        )

        # Sampling
        trace = sample(
            mcmc_params.n_mcmc_samples, step, start=trace[-1], 
            random_seed=mcmc_params.seed, progressbar=False
        )

    trace_b = np.array(trace['b'])
    if hparams['standardize']:
        if hparams['causality'] == [1, 2]:
            trace_b *= scale_ratio
        elif hparams['causality'] == [2, 1]:
            trace_b /= scale_ratio
        else:
            raise ValueError("Invalid value of causality: %s" %
                hparams['causality'])

    return {'b': trace_b}

def _is_uniform(L_cov_21):
    if type(L_cov_21) is not str:
        return False
    elif parse('U({:f},{:f})', L_cov_21.replace(' ', '')) is not None:
        return True
    else:
        raise ValueError("Invalid value of L_cov_21: %s" % L_cov_21)

def _get_L_cov(L_cov_21, floatX, Uniform, tt):
    if type(L_cov_21) in (float, int):
        return np.array([[1.0, L_cov_21], 
                         [L_cov_21, 1.0]]).astype(floatX)
    elif _is_uniform(L_cov_21):
        r = parse('U({:f},{:f})', L_cov_21.replace(' ', ''))
        L_cov_21_ = Uniform('L_cov_21_', lower=r[0], upper=r[1])
        return tt.stack([1.0, L_cov_21_, L_cov_21_, 1.0]).reshape((2, 2))

def _noise_model(hparams, h1, h2, xs_, Laplace, floatX, gamma):
    u"""Distribution of observation noise. 
    """
    # ---- Noise model ----
    if hparams['dist_noise'] == 'laplace':
        obs1 = lambda mu: Laplace(
            'x1s', mu=mu, b=h1 / np.float32(np.sqrt(2.)), 
            observed=xs_[:, 0], dtype=floatX
        )
        obs2 = lambda mu: Laplace(
            'x2s', mu=mu, b=h2 / np.float32(np.sqrt(2.)), 
            observed=xs_[:, 1], dtype=floatX
        )

    elif hparams['dist_noise'] == 'gg':
        beta = hparams['beta_noise']
        obs1 = lambda mu: GeneralizedGaussian(
            'x1s', mu=mu, beta=np.float32(beta), cov=np.eye(1), 
            observed=xs_[:, 0], dtype=floatX
        )
        obs2 = lambda mu: GeneralizedGaussian(
            'x2s', mu=mu, beta=np.float32(beta), cov=np.eye(1), 
            observed=xs_[:, 1], dtype=floatX
        )

    else:
        raise ValueError("Invalid value of dist_noise: %s" % 
            hparams['dist_noise'])

    return obs1, obs2

def _causal_effect(
    hparams, mu1, mu1s_, mu2, mu2s_, tau_cmmn, obs1, obs2, Normal, floatX):
    u"""Distribution of observations.
    """
    if hparams['causality'] == [1, 2]:
        # ---- Model 1: x1 -> x2 ----
        x1s = obs1(mu=mu1 + mu1s_)
        b = Normal('b', mu=np.float32(0.), 
                   tau=np.float32(1 / tau_cmmn[1]), dtype=floatX)
        x2s = obs2(mu=mu2 + mu2s_ + b * (x1s - mu1 - mu1s_)) \
              if hparams['subtract_mu_reg'] else \
              obs2(mu=mu2 + mu2s_ + b * x1s)

    else:
        # ---- Model 2: x2 -> x1 ----
        x2s = obs2(mu=mu2 + mu2s_)
        b = Normal('b', mu=np.float32(0.), 
                   tau=np.float32(1 / tau_cmmn[0]), dtype=floatX)
        x1s = obs1(mu=mu1 + mu1s_ + b * (x2s - mu2 - mu2s_)) \
              if hparams['subtract_mu_reg'] else \
              obs1(mu=mu1 + mu1s_ + b * x2s)

    return x1s, x2s, b

def _common_interceptions(hparams, tau_cmmn, Normal, floatX, verbose):
    if True: # hparams['fix_mu_zero']: debug
        mu1 = np.float32(0.0)
        mu2 = np.float32(0.0)

        if 10 <= verbose:
            print('Fix bias parameters to 0.0')

    else:
        if hparams['prior_var_mu'] == 'auto':
            tau1 = np.float32(1. / tau_cmmn[0])
            tau2 = np.float32(1. / tau_cmmn[1])
        else:
            v = hparams['prior_var_mu']
            tau1 = np.float32(1. / v)
            tau2 = np.float32(1. / v)
        mu1 = Normal('mu1', mu=np.float32(0.), tau=np.float32(tau1), 
                     dtype=floatX)
        mu2 = Normal('mu2', mu=np.float32(0.), tau=np.float32(tau2), 
                     dtype=floatX)

        if 10 <= verbose:
            print('mu1.dtype = {}'.format(mu1.dtype))
            print('mu2.dtype = {}'.format(mu2.dtype))

    return mu1, mu2

def _noise_variance(
    hparams, tau_cmmn, HalfNormal, Lognormal, Uniform, floatX, verbose):
    if hparams['prior_scale'] == 'tr_normal':
        h1 = HalfNormal('h1', tau=np.float32(1 / tau_cmmn[0]), dtype=floatX)
        h2 = HalfNormal('h2', tau=np.float32(1 / tau_cmmn[1]), dtype=floatX)

        if 10 <= verbose:
            print('Truncated normal for prior scales')

    elif hparams['prior_scale'] == 'log_normal':
        h1 = Lognormal('h1', tau=np.float32(1 / tau_cmmn[0]), dtype=floatX)
        h2 = Lognormal('h2', tau=np.float32(1 / tau_cmmn[1]), dtype=floatX)

        if 10 <= verbose:
            print('Log normal for prior scales')

    elif hparams['prior_scale'] == 'uniform':
        h1 = Uniform('h1', upper=np.float32(1 / tau_cmmn[0]), dtype=floatX)
        h2 = Uniform('h2', upper=np.float32(1 / tau_cmmn[1]), dtype=floatX)

        if 10 <= verbose:
            print('Uniform for prior scales')

    else:
        raise ValueError("Invalid value of prior_scale: %s" % 
            hparams['prior_scale'])

    return h1, h2

def _indvdl_t(
    hparams, std_x, n_samples, L_cov, StudentT, Deterministic, floatX, 
    cholesky, tt, verbose):
    df_L = hparams['df_indvdl']
    scale1 = np.float32(std_x[0] * hparams['v_indvdl_1'] / 
                        np.sqrt(df_L / (df_L - 2)))
    scale2 = np.float32(std_x[1] * hparams['v_indvdl_2'] / 
                        np.sqrt(df_L / (df_L - 2)))

    u1s = StudentT('u1s', nu=np.float32(df_L), shape=(n_samples,), 
                   dtype=floatX)
    u2s = StudentT('u2s', nu=np.float32(df_L), shape=(n_samples,), 
                   dtype=floatX)

    L_cov_ = cholesky(L_cov).astype(floatX)
    tt.set_subtensor(L_cov_[0, :], L_cov_[0, :] * scale1, inplace=True)
    tt.set_subtensor(L_cov_[1, :], L_cov_[1, :] * scale2, inplace=True)
    mu1s_ = Deterministic('mu1s_', 
                          L_cov_[0, 0] * u1s + L_cov_[0, 1] * u2s)
    mu2s_ = Deterministic('mu2s_', 
                          L_cov_[1, 0] * u1s + L_cov_[1, 1] * u2s)

    if 10 <= verbose:
        print('StudentT for individual effect')
        print('u1s.dtype = {}'.format(u1s.dtype))
        print('u2s.dtype = {}'.format(u2s.dtype))

    return mu1s_, mu2s_

def _indvdl_gauss(
    hparams, std_x, n_samples, L_cov, Normal, Deterministic, floatX, 
    cholesky, tt, verbose):
    scale1 = np.float32(std_x[0] * hparams['v_indvdl_1'])
    scale2 = np.float32(std_x[1] * hparams['v_indvdl_2'])

    u1s = Normal(
        'u1s', mu=np.float32(0.), tau=np.float32(1.), 
        shape=(n_samples,), dtype=floatX
    )
    u2s = Normal(
        'u2s', mu=np.float32(0.), tau=np.float32(1.), 
        shape=(n_samples,), dtype=floatX
    )
    L_cov_ = cholesky(L_cov).astype(floatX)
    tt.set_subtensor(L_cov_[0, :], L_cov_[0, :] * scale1, inplace=True)
    tt.set_subtensor(L_cov_[1, :], L_cov_[1, :] * scale2, inplace=True)
    mu1s_ = Deterministic('mu1s_', 
                          L_cov[0, 0] * u1s + L_cov[0, 1] * u2s)
    mu2s_ = Deterministic('mu2s_', 
                          L_cov[1, 0] * u1s + L_cov[1, 1] * u2s)

    if 10 <= verbose:
        print('Normal for individual effect')
        print('u1s.dtype = {}'.format(u1s.dtype))
        print('u2s.dtype = {}'.format(u2s.dtype))

    return mu1s_, mu2s_

def _indvdl_gg(
    hparams, std_x, n_samples, L_cov, Normal, Gamma, Deterministic, sgn, gamma, 
    floatX, cholesky, tt, verbose):
    # Uniform distribution on sphere
    gs = Normal('gs', np.float32(0.0), np.float32(1.0), 
                shape=(n_samples, 2), dtype=floatX)
    ss = Deterministic('ss', gs + sgn(sgn(gs) + np.float32(1e-10)) * 
                             np.float32(1e-10))
    ns = Deterministic('ns', ss.norm(L=2, axis=1)[:, np.newaxis])
    us = Deterministic('us', ss / ns)

    # Scaling s.t. variance to 1
    n = 2 # dimension
    beta = np.float32(hparams['beta_coeff'])
    m = n * gamma(0.5 * n / beta) \
        / (2 ** (1 / beta) * gamma((n + 2) / (2 * beta)))
    L_cov_ = (np.sqrt(m) * cholesky(L_cov)).astype(floatX)

    # Scaling to v_indvdls
    scale1 = np.float32(std_x[0] * hparams['v_indvdl_1'])
    scale2 = np.float32(std_x[1] * hparams['v_indvdl_2'])
    tt.set_subtensor(L_cov_[0, :], L_cov_[0, :] * scale1, inplace=True)
    tt.set_subtensor(L_cov_[1, :], L_cov_[1, :] * scale2, inplace=True)

    # Draw samples
    ts = Gamma(
        'ts', alpha=np.float32(n / (2 * beta)), beta=np.float32(.5), 
        shape=n_samples, dtype=floatX
    )[:, np.newaxis]
    mus_ = Deterministic(
        'mus_', ts**(np.float32(0.5 / beta)) * us.dot(L_cov_)
    )
    mu1s_ = mus_[:, 0]
    mu2s_ = mus_[:, 1]

    if 10 <= verbose:
        print('GG for individual effect')
        print('gs.dtype = {}'.format(gs.dtype))
        print('ss.dtype = {}'.format(ss.dtype))
        print('ns.dtype = {}'.format(ns.dtype))
        print('us.dtype = {}'.format(us.dtype))
        print('ts.dtype = {}'.format(ts.dtype))

    return mu1s_, mu2s_

def get_pm3_model_bmlingam(xs, hparams, verbose, force_mu_indvdl_zero=False):
    u"""Get a set of PyMC3 model of Bayesian mixed LiNGAM. 

    :param xs: Observation data. 
    :type xs: ndarray, shape=(n_samples, 2), dtype=float

    :param hparams: Hyperparameters. 
    :type hparams: Dict

    :return: Set of PyMC nodes. 
    :rtype: Dict
    """
    # ---- Import PyMC3 modules when required ----
    from pymc3 import Normal, Laplace, StudentT, Model, HalfNormal, \
                      Deterministic, Gamma, Lognormal, Uniform
    from scipy.special import gamma
    from theano.tensor import sgn
    from theano import config
    from theano.tensor.slinalg import cholesky
    import theano.tensor as tt

    floatX = config.floatX
    if 10 <= verbose:
        print('floatX = {}'.format(floatX))

    n_samples = xs.shape[0]
    xs_ = xs.astype(floatX)

    # ---- Scaling parameters ----
    std_x = np.std(xs, axis=0).astype(floatX)
    max_c = np.float32(hparams['max_c'])
    tau_cmmn = np.array(
        [(std_x[0] * max_c)**2, (std_x[1] * max_c)**2]).astype(floatX)

    # ---- "Model" block ----
    with Model() as model_bmlingam:
        # Individual specific effects (mu1s_, mu2s_)
        prior_indvdl = hparams['prior_indvdl']
        L_cov = _get_L_cov(hparams['L_cov_21'], floatX, Uniform, tt)

        if prior_indvdl == 't':
            mu1s_, mu2s_ = _indvdl_t(hparams, std_x, n_samples, L_cov, StudentT, 
                                     Deterministic, floatX, cholesky, tt, verbose)

        elif prior_indvdl == 'gauss':
            mu1s_, mu2s_ = _indvdl_gauss(hparams, std_x, n_samples, L_cov, 
                                         Normal, Deterministic, floatX, cholesky, tt, verbose)

        elif prior_indvdl == 'gg':
            mu1s_, mu2s_ = _indvdl_gg(hparams, std_x, n_samples, L_cov, 
                                      Normal, Gamma, Deterministic, 
                                      sgn, gamma, floatX, cholesky, tt, verbose)

        else:
            raise ValueError(
                'Invalid value of prior_indvdl: %s' % prior_indvdl)

        # Noise variance
        h1, h2 = _noise_variance(hparams, tau_cmmn, HalfNormal, Lognormal, 
                                 Uniform, floatX, verbose)

        # Common interceptions
        mu1, mu2 = _common_interceptions(hparams, tau_cmmn, Normal, floatX, 
                                         verbose)

        # Noise model
        obs1, obs2 = _noise_model(hparams, h1, h2, xs_, Laplace, floatX, gamma)

        # Causal effect
        x1s, x2s, b = _causal_effect(hparams, mu1, mu1s_, mu2, mu2s_, 
                                     tau_cmmn, obs1, obs2, Normal, floatX)

    if 10 <= verbose:
        print('mu1s_.dtype = {}'.format(mu1s_.dtype))
        print('mu2s_.dtype = {}'.format(mu2s_.dtype))
        print('h1.dtype = {}'.format(h1.dtype))
        print('h2.dtype = {}'.format(h2.dtype))
        print('x1s.dtype = {}'.format(x1s.dtype))
        print('x2s.dtype = {}'.format(x2s.dtype))

    return model_bmlingam
