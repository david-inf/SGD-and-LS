# -*- coding: utf-8 -*-

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# def call_f(w0, X, y, args=()):
#     return logistic(w0, X, y, *args)


def logistic(w, X, y, lam):
    z = - y * np.dot(X, w)
    L = np.sum(np.logaddexp(0, z))  # loss
    O = 0.5 * np.linalg.norm(w) ** 2  # regularizer
    return L + lam * O


def logistic_der(w, X, y, lam):
    z = - y * np.dot(X, w)
    dL = np.dot(- y * sigmoid(z), X)  # loss derivative
    dO = w  # regularizer derivative
    return dL + lam * dO


def f_and_df(w, X, y, lam):
    # fun, jac = f_and_df()
    z = - y * np.dot(X, w)  # once for twice
    L = np.sum(np.logaddexp(0, z))  # loss
    O = 0.5 * np.linalg.norm(w) ** 2  # regularizer
    dL = np.dot(- y * sigmoid(z), X)  # loss derivative
    dO = w  # regularizer derivative
    return (L + lam * O,  # objective function
            dL + lam * dO)  # jacobian


def logistic_hess(w, X, y, lam):
    z = y * np.dot(X, w)
    D = np.diag(sigmoid(z) * sigmoid(- z))
    ddL = np.dot(np.dot(D, X).T, X)
    ddO = lam * np.eye(w.shape[0])
    return ddL + ddO
