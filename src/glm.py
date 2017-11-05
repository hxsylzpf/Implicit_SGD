#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from AI_SGD import *
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

    def fit(self, X_train, y_train, learning_rate_0, nb_iterations, gamma,
            optimization_method):
        """fit a glm to the training data with the specified method"""

        if self.method == 'ai_sgd':

            ai_sgd = AI_SGD(learning_rate_0, nb_iterations, gamma)
            self.thetas_average, self.thetas_all = \
            ai_sgd.run(y_train, X_train, self.distr, optimization_method)
            self.thetas_last = np.array(self.thetas_all[nb_iterations-1:].T,
                                        ndmin=1)
            self.method_params = ai_sgd.get_params()

        else:
            raise ValueError("Method of fitting Unknown""")

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
        
