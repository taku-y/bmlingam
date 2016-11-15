# -*- coding: utf-8 -*-

"""Function to infer the posterior of the regression coefficient. 
"""
# Author: Taku Yoshioka
# License: MIT

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.kde import gaussian_kde

from bmlingam import do_mcmc_bmlingam

def _conf95(kde, dist_space):
    """Find 95%% confidence interval.

    This function works under the assumption that dist_space covers almost all 
    support (>99%%) of the distribution function. 
    """
    dens = kde(dist_space)
    cuml = np.cumsum(dens)
    dist = cuml / cuml[-1]
    ix_min = np.argmin(np.abs(dist - 0.025))
    ix_max = np.argmin(np.abs(dist - 0.975))
    return dist_space[ix_min], dist_space[ix_max]

def infer_coef_posterior(xs, hparams, mcmc_params, varnames=None, 
                         causality_str=None, verbose=1, 
                         plot_figure=True, show_plot=True):
    if varnames is None:
        varnames = ['var1', 'var2']

    if causality_str is None:
        causality_str = 'not given'

    # Infer posterior
    trace_ = do_mcmc_bmlingam(xs, hparams, mcmc_params)
    trace = trace_['b'].reshape(-1)

    # Stats
    kde = gaussian_kde(trace)
    dist_space = np.linspace(np.min(trace), np.max(trace), 100)
    x_min, x_max = _conf95(kde, dist_space) # Credible interval
    m = np.mean(trace) # Posterior mean

    # Summary
    if 1 <= verbose:
        print('---- Variables %s and %s ----' % (
            varnames[0], varnames[1]))
        print('Inferred causality   : %s' % causality_str)
        print('Posterior mean       : %f' % m)
        print('95%% Credible interval: (%f, %f)' % (x_min, x_max))
        print('')

    print(plot_figure, show_plot)
    # Plot distribution of posterior samples
    if plot_figure:
        plt.plot(dist_space, kde(dist_space))
        plt.xlabel('$b$')
        plt.ylabel('$P(b)$')
        plt.title(causality_str)

        # Confidence interval
        y_min, y_max = kde(x_min), kde(x_max)
        plt.plot([x_min, x_min], [0, y_min], 'b--')
        plt.plot([x_max, x_max], [0, y_max], 'b--')

        if show_plot:
            plt.show()

    return trace
