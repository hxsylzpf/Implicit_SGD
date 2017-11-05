#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def h(X, Thetas, distr):

    """Compute the h function according to the distribution"""

    if distr == 'poisson':
        h = np.exp(np.dot(X, Thetas))
    else:
        raise ValueError('Distribution unknown')

    return h


def log_likelihood(Y, X, thetas, distr):

    """We wcompute the log-likelihood as we want to find the MLE"""

    log_like = np.dot(Y.T, np.dot(X, thetas)) - np.sum(h(X, thetas, distr))
    return(log_like)


def rmse(y_predict, y):

    """We use RMSE as our loss"""

    return(np.sqrt(np.mean(np.square(y_predict.ravel()-y.ravel()))))


def metrics_evolution(thetas_all, Y, X, nb_points, metric, distr):

    """Compute the evolution of the metrics as we precise the values of thetas

    nb_points = size of the matrix we will return.
    This matrix will contain the metric calculated at every
    nb_iterations//nb_points iteration of the method, and the indices of these
    iterations

    returns:
    metric_evolution, a (nb_points x 2) matrix containing in the first column
    the indices of iteration and in the second the corresponding metric"""

    nb_iterations = len(thetas_all)

    if nb_points > nb_iterations:
        raise ValueError("Number of calculation asked too high""")

    else:

        metric_evolution = np.zeros((1, 2))

        for k in range(nb_points):

            indice = k * (nb_iterations//nb_points)
            thetas_k = np.take(thetas_all, indice, axis=0)

            if metric == 'rmse':
                y_predict_k = h(X, thetas_k, distr)
                metric_k = rmse(y_predict_k, Y)

            elif metric == 'log_likelihood':
                metric_k = log_likelihood(Y, X, thetas_k, distr)

            else:
                raise ValueError("Metric unknown")

            metric_evolution = np.append(metric_evolution,
                                         np.array([[indice, metric_k]]),
                                         axis=0)

    return(metric_evolution[1:])


def gini(y):

    """Compute the gini coefficient of the vector y"""

    bins = np.linspace(0., 100., 11)
    total = float(np.sum(y))
    yvals = []

    for b in bins:

        bin_vals = y[y <= np.percentile(y, b)]
        bin_fraction = (np.sum(bin_vals) / total) * 100.0
        yvals.append(bin_fraction)

    # perfect equality area
    pe_area = np.trapz(bins, x=bins)
    # lorenz area
    lorenz_area = np.trapz(yvals, x=bins)
    gini_val = (pe_area - lorenz_area) / float(pe_area)

    return bins, yvals, gini_val
