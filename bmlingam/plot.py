# -*- coding: utf-8 -*-

"""Plotting functions.
"""
# Author: Taku Yoshioka
# License: MIT

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats.kde import gaussian_kde
import time

from bmlingam.tests.usr_distrib import dist_names, usr_distrib
from bmlingam.tests.gendata import _gendata_latents as gendata_latents

def plot_usr_distrib():
    """Plot 19 distributions in usr_distrib.

    This program is used to check artificial data. 
    """
    dists = dist_names
    n_samples = 10000

    plt.figure(figsize=(15, 10.5))
    gs = gridspec.GridSpec(7, 3)
    xs = np.arange(-3, 3, 0.02)
    rng = np.random.RandomState(0)

    for i in xrange(19):
        ax = plt.subplot(gs[i])
        ys = usr_distrib(dists[i], 'pdf', xs, rng)
        k = usr_distrib(dists[i], 'kurt')
        samples = usr_distrib(dists[i], 'rnd', n_samples, rng)

        ax.plot(xs, ys)
        ax.set_ylim(0, np.max(ys) * 1.4)
        ax.text(-2.8, np.max(ys) * .9, 'k=%1.2f' % k, size=15)
        ax.hist(
            samples, bins=np.arange(-3, 3, 0.1), normed=1, facecolor='g', 
            alpha=0.8)

    # ---- Program finished ----
    print('Program finished at %s' % time.strftime("%c"))

def plot_gendata():
    """Plot artificial data with n_confounders=[0, 1, 6, 12].

    This program is used to check artificial data. 
    """
    n_samples = 200
    rng = np.random.RandomState(0)
    plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 2)

    # ---- Loop over the number of confonders ----
    for i, n_confounders in enumerate([0, 1, 6, 12]):
        # ---- Generate samples ----
        xs = gendata_latents(n_confounders, n_samples, rng)

        # ---- Plot samples ----
        ax = plt.subplot(gs[i])
        ax.scatter(xs[:, 0], xs[:, 1])
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_title('n_confounders=%d' % n_confounders)

    return

def plot_kde(ax, samples):
    """Plot kernel density estimation of given 1d samples.

    :param ax: Axes of matplotlib.
    :param samples: Samples.
    :type samples: numpy.ndarray, shape=(n_samples,)
    """
    kde = gaussian_kde(samples)
    dist_space = np.linspace(np.min(samples), np.max(samples), 100)
    ax.plot(dist_space, kde(dist_space))
