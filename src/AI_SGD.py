#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import optimize
import time

# We suppose that the first column of the X matrix is made of ones


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
