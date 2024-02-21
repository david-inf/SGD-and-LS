# -*- coding: utf-8 -*-

import numpy as np


def sigmoid(z):
    """ sigmoid function """
    return 1 / (1 + np.exp(-z))


def logistic(w, X, y, lam):
    """ Log-loss with l2 regularization """
    z = - y * np.dot(X, w)
    N = y.size  # number of samples

    # loss function
    loss = np.sum(np.logaddexp(0, z)) / N

    # regularizer term
    regul = 0.5 * np.linalg.norm(w) ** 2

    return loss + lam * regul


def logistic_der(w, X, y, lam):
    """ Log-loss with l2 regularization derivative """
    z = - y * np.dot(X, w)
    N = y.size  # number of samples

    # loss function derivative
    loss_der = np.dot(- y * sigmoid(z), X) / N

    # regularizer term derivative
    regul_der = w

    return loss_der + lam * regul_der


def f_and_df(w, X, y, lam):
    """ Log-loss with l2 regularization and its derivative """
    z = - y * np.dot(X, w)  # once for twice
    N = y.size  # number of samples

    # loss function and regularizer term
    loss = np.sum(np.logaddexp(0, z)) / N
    regul = 0.5 * np.linalg.norm(w) ** 2

    # loss function and regularizer term derivatives
    loss_der = np.dot(- y * sigmoid(z), X) / N
    regul_der = w

    return (loss + lam * regul,          # objective function
            loss_der + lam * regul_der)  # jacobian


def logistic_hess(w, X, y, lam):
    """ Log-loss with l2 regularization hessian """
    z = y * np.dot(X, w)
    N = y.size  # number of samples

    # diagnonal matrix NxN
    D = np.diag(sigmoid(z) * sigmoid(- z))

    # loss function hessian
    loss_hess = np.dot(np.dot(D, X).T, X) / N

    # regularizer term hessian
    regul_hess = np.eye(w.shape[0])

    return loss_hess + lam * regul_hess


# def call_f(w0, X, y, args=()):
#     return logistic(w0, X, y, *args)
