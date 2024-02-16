# -*- coding: utf-8 -*-

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic(w, X, y, lam):
    z = - y * np.dot(X, w)
    loss = np.sum(np.logaddexp(0, z))  # loss
    regul = 0.5 * np.linalg.norm(w) ** 2  # regularizer
    return loss + lam * regul


def logistic_der(w, X, y, lam):
    z = - y * np.dot(X, w)
    loss_der = np.dot(- y * sigmoid(z), X)  # loss derivative
    regul_der = w  # regularizer derivative
    return loss_der + lam * regul_der


def f_and_df(w, X, y, lam):
    # fun, jac = f_and_df()
    z = - y * np.dot(X, w)  # once for twice
    loss = np.sum(np.logaddexp(0, z))  # loss
    regul = 0.5 * np.linalg.norm(w) ** 2  # regularizer
    loss_der = np.dot(- y * sigmoid(z), X)  # loss derivative
    regul_der = w  # regularizer derivative
    return (loss + lam * regul,  # objective function
            loss_der + lam * regul_der)  # jacobian


def logistic_hess(w, X, y, lam):
    z = y * np.dot(X, w)
    D = np.diag(sigmoid(z) * sigmoid(- z))
    ddL = np.dot(np.dot(D, X).T, X)
    ddO = lam * np.eye(w.shape[0])
    return ddL + ddO


# def call_f(w0, X, y, args=()):
#     return logistic(w0, X, y, *args)
