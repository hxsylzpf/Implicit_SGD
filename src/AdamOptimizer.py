#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops


def create_mini_batches(X, Y, mini_batch_size=64):

    """Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, shape = (n_features, n_samples)
    Y -- output vector of shape (1, n_samples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    n_samples = X.shape[1]
    mini_batches = []

    # Shuffle the indices of X and Y
    permutation = list(np.random.permutation(n_samples))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, n_samples))

    # Create the partitions according to the shuffling
    num_complete_minibatches = math.floor(n_samples/mini_batch_size)

    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[:, k*mini_batch_size: min((k+1)*mini_batch_size, n_samples)]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size: min((k+1)*mini_batch_size, n_samples)]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def create_placeholders(n_features):

    """Creation of the placeholders for the tf session"""

    X = tf.placeholder(tf.float32, shape=[n_features, None])
    Y = tf.placeholder(tf.float32, shape=[None,1])

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


def forward_propagation(X, parameters):

    """Forward propagation for the perceptron

    Arguments:
    X -- input dataset placeholder, of shape (nb_features, nb_samples)
    parameters -- dictionary containing the tensor thetas

    Returns:
    y_predict -- the output of the perceptron"""

    thetas = parameters['thetas']
    y_predict = tf.exp(tf.matmul(thetas, X))
    return y_predict


def compute_loss(y_predict, y, n_samples):

    """Compute the rmse between the prediction and the true values"""

    cost = tf.reduce_sum(tf.pow(y_predict - y, 2))/(n_samples)
    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()
    (n_features, n_samples) = X_train.shape
    costs = []

    X, Y = create_placeholders(n_features)
    parameters = initialize_parameters(n_features)
    y_predict = forward_propagation(X, parameters)
    cost = compute_loss(y_predict, Y, n_samples)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_epochs):

            """epoch_cost = 0."""
            for (x, y) in zip(X_train.T, Y_train):
                sess.run([optimizer, cost],feed_dict={X: x, Y: y})
                
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            costs.append(c)
            """num_minibatches = int(n_samples / minibatch_size)
            minibatches = create_mini_batches(X_train, Y_train,
                                              minibatch_size)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost],
                                             feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost is True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost is True and epoch % 5 == 0:
                costs.append(epoch_cost)"""

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        return parameters
