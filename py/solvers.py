# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:15:19 2024

@author: Utente
"""

import numpy as np
from scipy.optimize import minimize
# from numba import jit


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


def predict(X, w, thresh=0.5):
    y_proba = sigmoid(np.dot(X, w))
    y_proba[y_proba > thresh] = 1
    y_proba[y_proba <= thresh] = -1
    return y_proba


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


def l_bfgs_b(w0, X, y, coeff):
    """
    Parameters
    ----------
    w0 : vector
        initial guess for optimal solution
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    """
    res = minimize(logistic, w0, args=(X, y, coeff), method="L-BFGS-B",
                   jac=logistic_der, bounds=None, options={"gtol": 1e-4})
    return res


def minibatch_gd_fixed(w0, alpha, M, X, y, coeff):
    # fun and grad are callable
    epochs = 300
    tol = 1e-4
    N, p = X.shape  # number of examples and features
    w_seq = np.zeros((epochs + 1, p))  # weights sequence, w\in\R^p
    w_seq[0,:] = w0
    fun_seq = np.zeros(epochs + 1)  # loss values sequence
    fun_seq[0] = logistic(w0, X, y, coeff)  # full loss
    grad_seq = np.zeros(epochs + 1)  # gradients norm sequence
    grad_seq[0] = np.linalg.norm(logistic_der(w0, X, y, coeff))  # full gradient
    k = 0  # epochs counter
    while grad_seq[k] > tol * (1 + np.absolute(fun_seq[k])) and k < epochs:
        ## Shuffle dataset
        batch = np.arange(N)  # dataset indices, reset every epoch
        rng = np.random.default_rng(k)  # set different seed every epoch
        rng.shuffle(batch)  # shuffle indices
        minibatches = np.array_split(batch, N / M)  # create the minibatches
        ## Approximate gradient and update (internal) weights
        y_seq = np.zeros((len(minibatches) + 1, p))  # internal weights sequence
        y_seq[0,:] = w_seq[k]
        for t in range(len(minibatches)):  # for minibatch in minibatches
        # for t, minibatch in enumerate(minibatches):
            ## Evaluate gradient approximation
            mini_grad = np.zeros(p)  # store true gradient approximation
            for j in minibatches[t]:
            # for j in minibatch:  # for index in minibatch indeces
                # evaluate gradient on a single example j at weight t
                mini_grad += logistic_der(y_seq[t], X[j,:], y[j], coeff)
            mini_grad = mini_grad / M
            ## Update (internal) weigths
            y_tnext = y_seq[t,:] - alpha * mini_grad  # model internal update
            y_seq[t+1,:] = y_tnext  # internal weights update
        ## Update sequence
        k += 1
        w_seq[k,:] = y_tnext
        fun_seq[k] = logistic(y_tnext, X, y, coeff)
        grad_seq[k] = np.linalg.norm(logistic_der(y_tnext, X, y, coeff))
    message = ""
    if grad_seq[-1] <= tol * (1 + np.absolute(fun_seq[-1])):
        message += "Gradient under tolerance"
    if k >= epochs:
        message += "Max epochs exceeded"
    # return w_seq[k,:]
    return w_seq[k,:], grad_seq[k], fun_seq[k], message


def minibatch_gd_decreasing(w0, alpha0, M, X, y, coeff):
    epochs = 300
    tol = 1e-4
    N, p = X.shape  # number of examples and features
    w_seq = np.zeros((epochs + 1, p))  # weights sequence, w\in\R^p
    w_seq[0,:] = w0
    fun_seq = np.zeros(epochs + 1)  # loss values sequence
    fun_seq[0] = logistic(w0, X, y, coeff)  # full loss
    grad_seq = np.zeros(epochs + 1)  # gradients norm sequence
    grad_seq[0] = np.linalg.norm(logistic_der(w0, X, y, coeff))  # full gradient
    k = 0  # epochs counter
    while grad_seq[k] > tol * (1 + np.absolute(fun_seq[k])) and k < epochs:
        ## Shuffle dataset
        batch = np.arange(N)  # dataset indices, reset every epoch
        rng = np.random.default_rng(k)  # set different seed every epoch
        rng.shuffle(batch)  # shuffle indices
        minibatches = np.array_split(batch, N / M)  # create the minibatches
        ## Approximate gradient and update (internal) weights
        y_seq = np.zeros((len(minibatches) + 1, p))  # internal weights sequence
        y_seq[0,:] = w_seq[k]
        for t in range(len(minibatches)):  # for minibatch in minibatches
        # for t, minibatch in enumerate(minibatches):
            ## Evaluate gradient approximation
            mini_grad = np.zeros(p)  # store true gradient approximation
            for j in minibatches[t]:
            # for j in minibatch:  # for index in minibatch indeces
                # evaluate gradient on a single example j at weight t
                mini_grad += logistic_der(y_seq[t], X[j,:], y[j], coeff)
            mini_grad = mini_grad / M
            ## Update (internal) weigths
            y_tnext = y_seq[t,:] - (alpha0 / (k + 1)) * mini_grad  # model internal update
            y_seq[t+1,:] = y_tnext  # internal weights update
        ## Update sequence
        k += 1
        w_seq[k,:] = y_tnext
        fun_seq[k] = logistic(y_tnext, X, y, coeff)
        grad_seq[k] = np.linalg.norm(logistic_der(y_tnext, X, y, coeff))
    message = ""
    if grad_seq[-1] <= tol * (1 + np.absolute(fun_seq[-1])):
        message += "Gradient under tolerance"
    if k >= epochs:
        message += "Max epochs exceeded"
    # return w_seq[k,:]
    return w_seq[k,:], grad_seq[k], fun_seq[k], message
