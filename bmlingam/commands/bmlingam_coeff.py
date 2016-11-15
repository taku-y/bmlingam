# -*- coding: utf-8 -*-

"""Include functions used in command bin/bmlingam-coeff. 
"""
# Author: Taku Yoshioka
# License: MIT

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np

from bmlingam import load_pklz, MCMCParams, infer_coef_posterior

def parse_args_bmlingam_coeff(args=None):
    parser = ArgumentParser()

    # Positional arguments
    parser.add_argument('optmodel_file', type=str, 
                        help='Optimal model file.')

    # Positional arguments
    parser.add_argument('--plot_figure', 
                        dest='is_plot_figure', action='store_true', 
                        help="""If this option is choosen, 
                                a plot of the posterior samples 
                                will be plotted.""")
    parser.add_argument('--no_plot_figure', 
                        dest='is_plot_figure', action='store_false', 
                        help="""If this option is choosen (default), 
                                a plot of the posterior samples 
                                will not be plotted.""")
    parser.set_defaults(is_plot_figure=False)

    parser.add_argument('--save_figure', 
                        dest='is_save_figure', action='store_true', 
                        help="""If this option is choosen (default), 
                                a plot of the posterior samples 
                                will be saved into a file.""")
    parser.add_argument('--no_save_figure', 
                        dest='is_save_figure', action='store_false', 
                        help="""If this option is choosen, 
                                a plot of the posterior samples 
                                will not be saved.""")
    parser.set_defaults(is_save_figure=True)

    parser.add_argument('--save_posterior', 
                        dest='is_save_posterior', action='store_true', 
                        help="""If this option is choosen (default), 
                                the posterior samples 
                                will be saved into a file.""")
    parser.add_argument('--no_save_posterior', 
                        dest='is_save_posterior', action='store_false', 
                        help="""If this option is choosen, 
                                the posterior samples 
                                will not be saved.""")
    parser.set_defaults(is_save_posterior=True)

    parser.add_argument('--figtype', 
                        dest='fig_ext', type=str, 
                        choices=['pdf', 'png'], 
                        help="""Figure file type (pdf or png).
                                Default is png. """)
    parser.set_defaults(fig_ext='png')

    # Get default setting
    default = MCMCParams()

    parser.add_argument('--n_mcmc_samples', 
                        default=default.n_mcmc_samples, type=int, 
                        help="""The number of MCMC samples (after burn-in).  
                                Default is {}. 
                            """.format(default.n_mcmc_samples))

    parser.add_argument('--n_burn', 
                        default=default.n_burn, type=int, 
                        help="""The number of burn-in samples in MCMC.  
                                Default is {}.
                            """.format(default.n_burn))

    parser.add_argument('--seed', 
                        default=default.seed, type=int, 
                        help="""Specify the seed of random number generator used in 
                                posterior sampling by MCMC. Default is {}.
                            """.format(default.seed))

    parser.add_argument('--seed_burn', 
                        default=default.seed_burn, type=int, 
                        help="""Specify the seed of random number generator used in 
                                the burn-in period of posterior sampling. 
                                Default is {}.
                            """.format(default.seed_burn))

    args_ = parser.parse_args(args)

    params = {
        'optmodel_file': args_.optmodel_file, 
        'mcmc_params': MCMCParams(
            n_mcmc_samples=args_.n_mcmc_samples, 
            n_burn=args_.n_burn, 
            seed=args_.seed, 
            seed_burn=args_.seed_burn, 
        ), 
        'is_plot_figure': args_.is_plot_figure, 
        'is_save_figure': args_.is_save_figure, 
        'is_save_posterior': args_.is_save_posterior, 
        'fig_ext': ('.' + args_.fig_ext)
    }

    return params

def bmlingam_coeff(
    optmodel_file, mcmc_params, is_plot_figure, is_save_figure, 
    is_save_posterior, fig_ext):
    """Estimate posterior distribution of regression coefficient.

    The bmlingam model is specified with optmodel_file. 
    """
    # Load optimal model info
    print(optmodel_file)
    optmodel = load_pklz(optmodel_file)
    xs = optmodel['xs']
    hparams = optmodel['hparams']
    causality_str = optmodel['causality_str']
    varnames = [optmodel['x1_name'], optmodel['x2_name']]

    # Plot parameters
    plot_figure = is_plot_figure or is_save_figure
    show_plot = is_plot_figure

    # Infer posterior
    trace = infer_coef_posterior(xs, hparams, mcmc_params, varnames, 
                                 causality_str, 1, plot_figure, show_plot)

    # Save plot
    if is_save_figure: 
        fig_file = optmodel_file.replace('.pklz', fig_ext)
        plt.savefig(fig_file)
        print('A figure of the distribution of the posterior samples' +
              'was saved as %s.' % fig_file)

    # Save posterior samples (MCMC trace)
    if is_save_posterior:
        csv_file = optmodel_file.replace('.pklz', '.post.csv')
        np.savetxt(csv_file, trace, delimiter=',')
        print('Posterior samples was saved as %s.' % csv_file)
