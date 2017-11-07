#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


def plot_gini(bins, result, gini_val):

    """Plot the gini curve"""

    plt.figure()
    plt.plot(bins, result, label="observed")
    plt.plot(bins, bins, '--', label="perfect eq.")
    plt.xlabel("fraction of population")
    plt.ylabel("fraction of wealth")
    plt.title("GINI: %.4f" % (gini_val))
    plt.legend()


def plot_loss(loss_test, loss_train, rmse):

    """plot the evolution of the loss over the train and test sets"""

    x_indices = loss_test[:, 0]
    y_loss_test = loss_test[:, 1]
    y_loss_train = loss_train[:, 1]

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x_indices, y_loss_train, label="training loss", c='blue')
    plt.ylabel("rmse")
    plt.yscale('log')
    plt.title("RMSE: %.4f \n RMSE evolution during the training" % (rmse))
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x_indices, y_loss_test, label="test loss", c='red')
    plt.xlabel("indice of iteration")
    plt.yscale('log')
    plt.legend()


def plot_log_likelihood(log_likelihood):

    """plot the evolution of the log_likelihood during the training"""

    x_indices = log_likelihood[:, 0]
    y_log_likelihood = log_likelihood[:, 1]

    plt.figure()
    plt.plot(x_indices, y_log_likelihood)
    plt.yscale('symlog')
    plt.xlabel("indice of iteration")
    plt.ylabel("log likelihood")
    plt.title("Log likelihood evolution during the training")


def plot_distributions(y_predict, y_test):

    """plot the histograms of y_predict and y_test"""

    max_y = max(max(y_predict), max(y_test))
    min_y = min(min(y_predict), min(y_test))
    range_y = (min_y, max_y)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.hist(y_predict, bins=50, range=range_y,
             label="y_predict", color='orange')
    plt.ylabel("nb of contracts")
    plt.yscale('log')
    plt.title("Distribution predicted vs real distribution")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.hist(y_test, range=range_y,
             label="y_test", color='green')
    plt.xlabel("nb of damages")
    plt.yscale('log')
    plt.legend()
