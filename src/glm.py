#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import optimizer
import metrics


class GLM(object):
    """Class that uses two methods for estimating Generalized Linear Models.
    The User have the choice to use:
        - the Average implicit Stochastic Gradient Descent for GLM
        (from Ptoulis, 2016)
        - the Adam Optimizer from tensorflow (the GLM is then encoded
        as a perceptron with the 'distr' used as the activation function)"""

    def __init__(self, distr='poisson', method='ai_sgd'):

        self.distr = distr
        self.method = method
        self.method_params = None
        self.thetas_average = None
        self.thetas_all = None
        self.thetas_last = None
        self.loss_train = None
        self.loss_test = None
        self.log_likelihood = None
        self.rmse = None
        self.gini = None

    def get_params(self):
        """returns the parameters of the glm estimation as a dict"""

        return dict(
                (
                        ('distr', self.distr),
                        ('method', self.method),
                        ('method_params', self.method_params),
                        ('thetas_average', self.thetas_average),
                        ('thetas_all', self.thetas_all),
                        ('thetas_last', self.thetas_last),
                        ('loss_train', self.loss_train),
                        ('loss_test', self.loss_test),
                        ('log_likelihood', self.log_likelihood),
                        ('rmse', self.rmse),
                        ('gini', self.gini),
                )
                    )

    def fit(self, X_train, y_train, learning_rate, **kwargs):

        """fit a glm to the training data with the specified method"""

        if self.method == 'ai_sgd':

            nb_iterations = kwargs.get('nb_iterations', None)
            gamma = kwargs.get('gamma', None)
            optimization_method = kwargs.get('optimization_method', None)

            Optimizer = optimizer.AI_SGD(learning_rate, nb_iterations, gamma)
            self.thetas_average, self.thetas_all = \
                Optimizer.run(y_train, X_train, self.distr,
                              optimization_method)

        elif self.method == 'AdamOptimizer':

            nb_epochs = kwargs.get('nb_epochs', None)
            Optimizer = optimizer.AdamOptimizer(learning_rate, nb_epochs)
            self.thetas_all = Optimizer.run(X_train.T, y_train.T, self.distr)

        else:
            raise ValueError("Method of fitting Unknown""")

        #self.thetas_last = real_thetas
        self.thetas_last = np.array(self.thetas_all[-1].T, ndmin=1)
        self.method_params = Optimizer.get_params()

    def predict(self, X_test):
        """apply the model learned after fitting to the test set"""

        return(metrics.h(X_test, self.thetas_last, self.distr))

    def compute_metrics(self, X_test, y_test):

        """Compute the final value of the metrics"""

        y_predict = metrics.h(X_test, self.thetas_last, self.distr)
        self.rmse = metrics.rmse(y_predict, y_test)
        gini_bins, gini_yvals, self.gini = metrics.gini(y_predict)
        duration_of_computation = self.method_params['time_taken']

        return(self.rmse, self.gini, gini_bins, gini_yvals,
               duration_of_computation)

    def compute_metrics_evolution(self, X_train, X_test,
                                  y_test, y_train, nb_points):
        """Compute the evolution of the metrics over the test and train
        sets as the thetas are more and more precises"""

        self.loss_train = metrics.metrics_evolution(self.thetas_all, y_train,
                                                    X_train, nb_points, 'rmse',
                                                    self.distr)
        self.loss_test = metrics.metrics_evolution(self.thetas_all, y_test,
                                                   X_test, nb_points, 'rmse',
                                                   self.distr)
        self.log_likelihood = metrics.metrics_evolution(self.thetas_all,
                                                        y_train, X_train,
                                                        nb_points,
                                                        'log_likelihood',
                                                        self.distr)

        return(self.log_likelihood, self.loss_train, self.loss_test)

  
real_thetas = np.array([-2.223848,
                        .156882,
                        1.056299,
                        -.8487041,
                        -.2053206,
                        .1231854,
                        -.4400609,
                        .0797984,
                        .1869484,
                        .1268465,
                        .030081,
                        .1140853,
                        .1411583])
       
