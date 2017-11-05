#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sps


class random_dataset(object):

    """Create random train and test sets which follow the probabilistic law
    wanted"""

    def __init__(self, n_features, distr):

        self.n_features = n_features
        self.distr = distr

        beta_0 = np.random.normal(0.0, 1.0, 1)
        beta = sps.rand(n_features, 1, 0.1)
        beta = np.array(beta.todense())
        self.beta = np.append(beta_0, beta)

    def create(self, n_samples):

        """Create a random sparse (n_samples x n_features+1 ) matrix and
        a Y vector of length n_samples which follows the given
        distribution

        The X matrix is created as follows:
            (1, feature_1,....feature_n_features)"""

        X = np.random.normal(0.0, 1.0, [n_samples, self.n_features])
        X = np.append(np.ones((n_samples, 1)), X, axis=1)

        if self.distr == 'poisson':
            y = np.exp(np.dot(X, self.beta))

        else:
            raise ValueError("Distribution Unknown")

        return(X, y)
