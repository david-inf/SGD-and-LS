# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:15:19 2024

@author: Utente
"""

import numpy as np
from scipy.optimize import minimize


def sigmoid(x):
    """
    Parameters
    ----------
    x : vector or scalar

    Returns
    -------
    vector or scalar like x.shape

    """
    return 1 / (1 + np.exp(-x))


def logistic(w, X, y, coeff):
    """
    Parameters
    ----------
    w : vector
    X : matrix or vector
    y : vector or scalar
    coeff : scalar

    Returns
    -------
    scalar
    """
    loss_vec = np.exp(- y * np.dot(X, w))
    loss = np.sum(loss_vec)
    regul = coeff * np.linalg.norm(w) ** 2
    return loss + regul


def logistic_der(w, X, y, coeff):
    """
    Parameters
    ----------
    w : vector
    X : matrix or vector
    y : vector or scalar
    coeff : scalar

    Returns
    -------
    vector like w.shape[0]
    """
    r = - y * sigmoid(- y * np.dot(X, w))
    loss_der = np.dot(r, X)
    regul_der = coeff * 2 * w
    return loss_der + regul_der


def logistic_hess(w, X, y, coeff):
    """
    Parameters
    ----------
    w : vector
    X : matrix or vector
    y : vector or scalar
    coeff : scalar

    Returns
    -------
    Matrix like (w.shape[0], w.shape[0])
    """
    if X.ndim == 2:
        D = np.diag(sigmoid(y * np.dot(X, w)) * sigmoid(- y * np.dot(X, w)))
    else:
        D = sigmoid(y * np.dot(X, w)) * sigmoid(- y * np.dot(X, w))
    loss_hess = np.dot(np.dot(D, X).T, X)
    regul_hess = 2 * coeff * np.eye(w.shape[0])
    return loss_hess + regul_hess


def l_bfgs_b(w0, X, y):
    res = minimize(logistic, w0, args=(X, y), method="L-BFGS-B", jac=logistic_der,
                   bounds=None, options={"gtol": 1e-4})
    return res
