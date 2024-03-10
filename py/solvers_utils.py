# -*- coding: utf-8 -*-

import numpy as np


def sigmoid(z):
    """ sigmoid function """

    return 1 / (1 + np.exp(-z))


def logistic(w, X, y, lam):
    """ Log-loss with l2 regularization """
    samples = y.size  # number of samples

    # loss function
    z = - y * np.dot(X, w)
    loss = np.sum(np.logaddexp(0, z)) / samples

    # regularizer term
    regul = 0.5 * np.linalg.norm(w) ** 2

    return loss + lam * regul


def loss_and_regul(w, X, y, lam):
    """ Log-loss and log-loss with l2 regularization term """
    samples = y.size  # number of samples

    # loss function
    z = - y * np.dot(X, w)
    loss = np.sum(np.logaddexp(0, z)) / samples

    # regularizer term
    regul = 0.5 * np.linalg.norm(w) ** 2

    return (loss,                # log-loss
            loss + lam * regul)  # log-loss with l2 regularization


def logistic_der(w, X, y, lam):
    """ Log-loss with l2 regularization derivative """
    samples = y.size  # number of samples

    # loss function derivative
    z = - y * np.dot(X, w)
    loss_der = np.dot(- y * sigmoid(z), X) / samples

    # regularizer term derivative
    regul_der = w

    return loss_der + lam * regul_der


def f_and_df_log(w, X, y, lam):
    """ Log-loss with l2 regularization and its derivative """
    samples = y.size  # number of samples

    z = - y * np.dot(X, w)  # once for twice

    # loss function and regularizer term
    loss = np.sum(np.logaddexp(0, z)) / samples
    regul = 0.5 * np.linalg.norm(w) ** 2

    # loss function and regularizer term derivatives
    loss_der = np.dot(- y * sigmoid(z), X) / samples
    regul_der = w

    return (loss + lam * regul,          # objective function
            loss_der + lam * regul_der)  # jacobian


def logistic_hess(w, X, y, lam):
    """ Log-loss with l2 regularization hessian """
    z = y * np.dot(X, w)
    samples = y.size  # number of samples

    # diagnonal matrix NxN
    D = np.diag(sigmoid(z) * sigmoid(- z))

    # loss function hessian
    loss_hess = np.dot(np.dot(D, X).T, X) / samples

    # regularizer term hessian
    regul_hess = np.eye(w.size)

    return loss_hess + lam * regul_hess




def linear(w, X, y, lam):
    """ Quadratic-loss with l2 regularization """
    samples = y.size  # number of samples

    # loss function
    # loss = 0.5 * np.linalg.norm(np.dot(X, w) - y)**2 / samples
    XXw = np.dot(np.matmul(X.T, X), w)
    yX = np.dot(y, X)
    loss = 0.5 * (np.dot(w, XXw) - 2 * np.dot(yX, w) + np.linalg.norm(y)**2) / samples

    # regularizer term
    regul = 0.5 * np.linalg.norm(w) ** 2

    return loss + lam * regul


def linear_der(w, X, y, lam):
    """ Quadratic-loss with l2 regularization derivative """
    samples = y.size  # number of samples

    # loss function derivative
    loss_der = np.dot(np.matmul(X.T, X), w) - np.dot(y, X) / samples

    # regularizer term derivative
    regul_der = w

    return loss_der + lam * regul_der


def f_and_df_linear(w, X, y, lam):
    """ Quadratic-loss with l2 regularization and its derivative """
    samples = y.size  # number of samples

    XXw = np.dot(np.matmul(X.T, X), w)  # once for twice
    yX = np.dot(y, X)  # once for twice

    # loss function and regularizer term
    

    # loss function and regularizer term derivatives
    

    return (loss + lam * regul,          # objective function
            loss_der + lam * regul_der)  # jacobian


def linear_hess(w, X, y, lam):
    """ Quadratic-loss with l2 regularization hessian """
    samples = y.size  # number of samples

    # loss function hessian
    loss_hess = np.matmul(X.T, X) / samples

    # regularizer term hessian
    regul_hess = np.eye(w.size)

    return loss_hess + lam * regul_hess
