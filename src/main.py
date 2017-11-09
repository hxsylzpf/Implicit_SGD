#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import glm
import create_random_dataset
import graphics
import file_cleaning
import racd3_data

import metrics
import matplotlib.pyplot as plt



# Creates the test and train sets

# Random creation
"""
dataset = create_random_dataset.random_dataset(n_features=100, distr='poisson')
X_train, y_train = dataset.create(n_samples=1000)
X_test, y_test = dataset.create(n_samples=1000)
y_train = y_train.reshape((len(y_train), 1))
y_test = y_test.reshape((len(y_test), 1))
print(y_train.shape, X_train.shape)"""


# MRH data
"""
dataset, y, exposure = file_cleaning.dataset_to_use()
n_samples = len(dataset)
X = np.append(np.ones((n_samples, 1)), dataset, axis=1)
X_train = X[:10000]
X_test = X[10000:]
y_train = y[:10000]
y_test = y[10000:]
print('dataset charged')"""

# racd3 data

X, y = racd3_data.load_racd3_data()
n_samples = len(y)
X = np.append(np.ones((n_samples, 1)), X, axis=1)
X_train = X[:3000]
X_test = X[3000:]
y_train = y[:3000]
y_test = y[3000:]
print('dataset charged')

# Fit a glm to the dataset

# AI_SGD

"""
Glm = glm.GLM(distr='poisson', method='ai_sgd')
Glm.fit(X_train, y_train, learning_rate=0.1, nb_iterations=1000, gamma=0.6,
        optimization_method='brenth')
y_predict = Glm.predict(X_test).ravel()
print('model fitted')"""

# Perceptron

Glm = glm.GLM(distr='poisson', method='AdamOptimizer')
Glm.fit(X_train, y_train, learning_rate=0.001, nb_epochs=40)
y_predict = Glm.predict(X_test).ravel()
print('model fitted')

# Compute the metrics to assess the method

log_likelihood, loss_train, loss_test = \
    Glm.compute_metrics_evolution(X_train, X_test, y_test, y_train,
                                  nb_points=100)

rmse, gini, gini_bins, gini_yvals, duration_of_computation = \
    Glm.compute_metrics(X_test, y_test)
print('metrics computed')

# Plot all the graphes necessary to evaluate the performances of our method

graphics.plot_loss(loss_test, loss_train, rmse)
graphics.plot_log_likelihood(log_likelihood)
graphics.plot_gini(gini_bins, gini_yvals, gini)
graphics.plot_distributions(y_predict, y_test)
print('The total computation time needed to fit the model is: ',
      duration_of_computation, ' seconds')

# real data plot

y_predict_real = metrics.h(X_test, glm.real_thetas,'poisson')
graphics.plot_distributions(y_predict_real, y_test)
print('real rmse: ', metrics.rmse(y_predict_real, y_test))
norm = np.linalg.norm(Glm.thetas_last-glm.real_thetas)
print('distance between the thetas = ', norm)