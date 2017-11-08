#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import optimize
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

# We suppose that the first column of the X matrix is made of ones

# ----GENERAL PURPOSES FUNCTIONS----


def shuffle_dataset(nb_iterations, m):

    """create a list of nb_iterations indices
    uniformly taken between [0,m-1]"""

    list_indices = np.random.randint(m, size=nb_iterations)
    return(list_indices)


def h(X_n, Thetas_n, distr):

    """Compute the h function according to the distribution"""

    if distr == 'poisson':
        h = np.exp(np.dot(X_n, Thetas_n))
    else:
        raise ValueError('Distribution unknown')

    return h

# ----FUNCTIONS FOR THE TENSORFLOW PERCEPTRON----


def create_placeholders():

    """Creation of the placeholders for the tf session"""

    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    return X, Y


def initialize_parameters(n_features):

    """Initializes the thetas to build a perceptron with tensorflow.
    The shape is: thetas : [1, n_features]

    Returns a dictionnary containing the tensor thetas"""

    thetas = tf.get_variable("thetas",
                             [1, n_features],
                             initializer=tf.contrib.layers.xavier_initializer(seed=1))
    parameters = {"thetas": thetas}

    return parameters


def forward_propagation(X, parameters, distr):

    """Forward propagation for the perceptron

    Arguments:
    X -- input dataset placeholder, of shape (nb_features, nb_samples)
    parameters -- dictionary containing the tensor thetas

    Returns:
    y_predict -- the output of the perceptron"""

    thetas = parameters['thetas']

    if distr == 'poisson':

        y_predict = tf.exp(tf.matmul(thetas, X))

    else:
        raise ValueError("Distribution Unknown")

    return y_predict


def compute_loss(y_predict, y):

    """Compute the rmse between the prediction and the true values"""

    cost = tf.sqrt(tf.reduce_mean(tf.pow(y_predict - y, 2)))
    return cost


def run_AdamOptimizer(X_train, Y_train, distr, learning_rate=0.0001,
                      nb_epochs=100):

    ops.reset_default_graph()
    (n_features, n_samples) = X_train.shape
    thetas_all = np.zeros((n_samples*nb_epochs, n_features))
    cpt = 0

    X, Y = create_placeholders()
    parameters = initialize_parameters(n_features)
    y_predict = forward_propagation(X, parameters, distr)
    cost = compute_loss(y_predict, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(nb_epochs):

            for (x, y) in zip(X_train.T, Y_train.T):

                y = y.reshape((1, 1))
                x = x.reshape((len(x), 1))

                sess.run(optimizer, feed_dict={X: x, Y: y})
                theta_dict = sess.run(parameters)
                theta_k = np.array(theta_dict['thetas']).ravel()
                thetas_all[cpt] = theta_k
                cpt += 1

        return thetas_all


class AdamOptimizer(object):

    def __init__(self, learning_rate=0.001, nb_epochs=10000):

        self.learning_rate = learning_rate
        self.nb_epochs = nb_epochs
        self.time_start = None
        self.time_end = None
        self.time_taken = None

    def get_params(self):
        """returns the parameters of the ai_sgd method as a dict"""
        return dict(
                (
                        ('learning_rate', self.learning_rate),
                        ('nb_epoch', self.nb_epochs),
                        ('time_start', self.time_start),
                        ('time_end', self.time_end),
                        ('time_taken', self.time_taken)
                )
            )

    def run(self, X_train, Y_train, distr):
        """run the ai sgd methods and outputs the theta vector"""
        self.time_start = time.time()
        Thetas_all = run_AdamOptimizer(X_train,
                                       Y_train,
                                       distr,
                                       self.learning_rate,
                                       self.nb_epochs)
        self.time_end = time.time()
        self.time_taken = self.time_end - self.time_start

        return(Thetas_all)


# ----FUNCTIONS FOR THE IMPLICIT SGD METHOD-----


def learning_rate(learning_rate_0, n, gamma):

    """The learning rate is defined as follows:
        gamma_n = learning_rate_0/(n^gamma) """

    return(learning_rate_0*((n+1)**-gamma))


def equation_to_solve(xi, learning_rate_n, Y_n, X_n, Thetas_n, distr):

    """Returns the function we have to set to zero to solve the implicit
    equation"""

    F = learning_rate_n * (Y_n - h(Thetas_n, X_n, distr) *
                           h(xi*X_n, X_n, distr)) - xi
    return(F)


def equation_to_solve_prime(xi, learning_rate_n, Y_n, X_n, Thetas_n, distr):

    """Returns the derivative (according to xi) of the function we have to
    set to zero to solve the implicit equation"""

    F_prime = -learning_rate_n * h(Thetas_n, X_n, distr) * \
        h(xi*X_n, X_n, distr) * np.dot(X_n, X_n) - 1
    return(F_prime)


def solve_implicit_equation(learning_rate_n, Y_n, X_n, Thetas_n, xi_0,
                            method, distr):

    """use an optimization method to find the root of the implicit equation
    at each iteration"""

    if method == 'brenth':

        if xi_0 > 0:

            res = optimize.brenth(f=equation_to_solve,
                                  a=0,
                                  b=xi_0,
                                  args=(learning_rate_n,
                                        Y_n,
                                        X_n,
                                        Thetas_n,
                                        distr)
                                  )
        else:

            res = optimize.brenth(f=equation_to_solve,
                                  a=xi_0,
                                  b=0,
                                  args=(learning_rate_n,
                                        Y_n,
                                        X_n,
                                        Thetas_n,
                                        distr)
                                  )

    else:

        raise ValueError('Optimization method unknown')

    return(res)


def init_variables(nb_iterations, p):

    """Create the variables r_n and Thetas.

    r_n = scalar which countains the bounds to the
    search of the solution of the implicit equation at each iteration n.

    Thetas = matrix of size (nb_iterations + 1) x p which countains
    the updates of the parameter theta (vector of size p)
    It is initialized by a vector of random numbers between 0 and 1"""

    r_n = 0
    theta_0 = np.random.random(size=(1, p))
    Thetas = np.append(theta_0, np.zeros((nb_iterations, p)), axis=0)

    return(r_n, Thetas)


def compute_r(Thetas_n, Y_n, X_n, learning_rate_n, distr):

    """Compute the bound r_n of the search for the solution to the implicit
    equation at the iteration n and adds it to the vector r"""

    r_n = learning_rate_n * (Y_n - h(X_n, Thetas_n, distr))
    return(r_n)


def update_Thetas(Thetas_n, update_rate, X_n):

    """Compute the update of the parameter theta at iteration n and adds it
    to the matrix Thetas"""

    new_Thetas = Thetas_n + update_rate*X_n
    return(new_Thetas)


def run_ai_sgd(learning_rate_0, Y_train, X_train, nb_iterations, gamma,
               distr, optimization_method):

    """Applying the average implicit sgd procedure to fit a glm"""

    (m, p) = X_train.shape
    (r_n, Thetas) = init_variables(nb_iterations, p)
    indices = shuffle_dataset(nb_iterations, m)

    for n in range(nb_iterations):

        X_n = np.take(X_train, indices[n], axis=0)
        Y_n = np.take(Y_train, indices[n], axis=0)
        Thetas_n = Thetas[n]

        learning_rate_n = learning_rate(learning_rate_0, n, gamma)
        r_n = compute_r(Thetas_n, Y_n, X_n, learning_rate_n, distr)
        update_rate = solve_implicit_equation(
                learning_rate_n,
                Y_n,
                X_n,
                Thetas_n,
                r_n,
                optimization_method,
                distr)
        Thetas[n+1] = update_Thetas(Thetas_n, update_rate, X_n)

    return(np.mean(Thetas[1:], axis=0), Thetas[1:])


class AI_SGD(object):

    def __init__(self, learning_rate_0=1, nb_iterations=10000, gamma=0.5):

        self.learning_rate = learning_rate_0
        self.nb_iterations = nb_iterations
        self.gamma = gamma
        self.time_start = None
        self.time_end = None
        self.time_taken = None

    def get_params(self):
        """returns the parameters of the ai_sgd method as a dict"""
        return dict(
                (
                        ('learning_rate', self.learning_rate),
                        ('nb_iterations', self.nb_iterations),
                        ('gamma', self.gamma),
                        ('time_start', self.time_start),
                        ('time_end', self.time_end),
                        ('time_taken', self.time_taken)
                )
            )

    def run(self, Y_train, X_train, distr, optimization_method):
        """run the ai sgd methods and outputs the theta vector"""
        self.time_start = time.time()
        thetas, Thetas_all = run_ai_sgd(self.learning_rate,
                                        Y_train,
                                        X_train,
                                        self.nb_iterations,
                                        self.gamma,
                                        distr,
                                        optimization_method)
        self.time_end = time.time()
        self.time_taken = self.time_end - self.time_start

        return(thetas, Thetas_all)
