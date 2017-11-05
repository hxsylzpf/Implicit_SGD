#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glm
import create_random_dataset
import graphics

# Create the test and train sets

dataset = create_random_dataset.random_dataset(n_features=500, distr='poisson')
X_train, y_train = dataset.create(n_samples=1000000)
X_test, y_test = dataset.create(n_samples=10000)

# Fit a glm to the dataset

Glm = glm.GLM(distr='poisson', method='ai_sgd')
Glm.fit(X_train, y_train, learning_rate_0=10, nb_iterations=20000, gamma=0.5,
        optimization_method='brenth')
y_predict = Glm.predict(X_test).ravel()

# Compute the metrics to assess the method

log_likelihood, loss_train, loss_test = \
    Glm.compute_metrics_evolution(X_train, X_test, y_test, y_train, 10)

rmse, gini, gini_bins, gini_yvals, duration_of_computation = \
    Glm.compute_metrics(X_test, y_test)

# Plot all the graphes necessary to evaluate the performances of our method

graphics.plot_loss(loss_test, loss_train, rmse)
graphics.plot_log_likelihood(log_likelihood)
graphics.plot_gini(gini_bins, gini_yvals, gini)
print('The total computation time needed to fit the model is: ',
      duration_of_computation, ' seconds')
