# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:15:19 2024

@author: Utente
"""

#%% Packages
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult
# from numba import jit
from my_utils import logistic, logistic_der, f_and_df, f_and_df_2

#%% Benchmark solver
def l_bfgs_b(w0, X, y):
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
    
    # res = minimize(logistic, w0, args=(X, y, coeff), method="L-BFGS-B",
    #                jac=logistic_der, bounds=None, options={"gtol": 1e-4})
    res = minimize(f_and_df_2, w0, args=(X, y), method="L-BFGS-B",
                   jac=True, bounds=None, options={"gtol": 1e-4})
    return res


#%% Solvers utils
def minibatch_gradient(X, y, minibatch, y_tbefore, p, M):
    """
    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    coeff : TYPE
        DESCRIPTION.
    minibatch : TYPE
        set of indices
    y_tbefore : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.
    M : TYPE
        DESCRIPTION.

    Returns
    -------
    mini_grad : TYPE
        DESCRIPTION.
    """
    mini_grad = np.zeros(p)  # store true gradient approximation
    # for j in minibatches[t]:
    for j in minibatch:  # for index in minibatch indeces
        # evaluate gradient on a single example j on a given weight
        mini_grad += logistic_der(y_tbefore, X[j, :], y[j])
    mini_grad = np.divide(mini_grad, M)
    return mini_grad


def shuffle_dataset(N, k, M):
    batch = np.arange(N)  # dataset indices, reset every epoch
    rng = np.random.default_rng(k)  # set different seed every epoch
    rng.shuffle(batch)  # shuffle indices
    # array_split is expensive
    minibatches = np.array_split(batch, N / M)  # create the minibatches
    return minibatches


#%% Minibatch Gradient Descent with fixed step-size
def minibatch_gd_fixed(w0, alpha, M, X, y):
    epochs = 200
    tol = 1e-4
    N, p = X.shape  # number of examples and features
    # weights sequence, w\in\R^p
    w_seq = np.zeros((epochs + 1, p))
    w_seq[0, :] = w0
    # full objective function and full gradient norm sequences
    fun_seq = np.zeros(epochs + 1)
    grad_seq = np.zeros(epochs + 1)
    fun_seq[0], grad_seq[0] = f_and_df(w0, X, y)
    k = 0  # epochs counter
    while grad_seq[k] > tol * (1 + fun_seq[k]) and k < epochs:
        ## Shuffle dataset
        minibatches = shuffle_dataset(N, k, M)
        ## Approximate gradient and update (internal) weights
        # internal weights sequence
        y_seq = np.zeros((len(minibatches) + 1, p))
        y_seq[0, :] = w_seq[k]
        # for t in range(len(minibatches)):  # for minibatch in minibatches
        for t, minibatch in enumerate(minibatches):
            ## Evaluate gradient approximation
            mini_grad = minibatch_gradient(X, y, minibatch, y_seq[t, :], p, M)
            ## Compute direction
            # d_t = - mini_grad
            ## Update (internal) weights
            y_tnext = y_seq[t, :] - alpha * mini_grad  # model internal update
            y_seq[t+1, :] = y_tnext  # internal weights update
        ## Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_tnext
        fun_seq[k], grad_seq[k] = f_and_df(y_tnext, X, y)
    message = ""
    if grad_seq[-1] <= tol * (1 + np.absolute(fun_seq[-1])):
        message += "Gradient under tolerance"
    if k >= epochs:
        message += "Max epochs exceeded"
    # return w_seq[k,:]
    return w_seq[k, :], grad_seq[k], fun_seq[k], message


#%% Minibatch Gradient Descent with decreasing step-size
def minibatch_gd_decreasing(w0, alpha0, M, X, y):
    epochs = 200
    tol = 1e-4
    N, p = X.shape  # number of examples and features
    # weights sequence, w\in\R^p
    w_seq = np.zeros((epochs + 1, p))
    w_seq[0, :] = w0
    # full objective function and full gradient norm sequences
    fun_seq = np.zeros(epochs + 1)
    grad_seq = np.zeros(epochs + 1)
    fun_seq[0], grad_seq[0] = f_and_df(w0, X, y)
    k = 0  # epochs counter
    while grad_seq[k] > tol * (1 + fun_seq[k]) and k < epochs:
        ## Shuffle dataset
        minibatches = shuffle_dataset(N, k, M)
        ## Approximate gradient and update (internal) weights
        # internal weights sequence
        y_seq = np.zeros((len(minibatches) + 1, p))
        y_seq[0, :] = w_seq[k]
        for t, minibatch in enumerate(minibatches):
            ## Evaluate gradient approximation
            mini_grad = minibatch_gradient(X, y, minibatch, y_seq[t, :], p, M)
            ## Update (internal) weigths
            alpha = alpha0 / (k + 1)
            y_tnext = y_seq[t, :] - alpha * mini_grad  # model internal update
            y_seq[t+1, :] = y_tnext  # internal weights update
        ## Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_tnext
        fun_seq[k], grad_seq[k] = f_and_df(y_tnext, X, y)
    message = ""
    if grad_seq[-1] <= tol * (1 + np.absolute(fun_seq[-1])):
        message += "Gradient under tolerance"
    if k >= epochs:
        message += "Max epochs exceeded"
    # return w_seq[k,:]
    return w_seq[k, :], grad_seq[k], fun_seq[k], message


#%% Minibatch Gradient Descent with Armijo line search
def reset_step(N, alpha, alpha0, M, a, t, opt):
    """
    Parameters
    ----------
    N : int
        Dataset examples
    alpha : float
        Previous iteration (mini-batch) step-size
    alpha0 : float
        Maximum step-size
    M : int
        Mini-batch size
    a : float
        arbitrary constant greater than 1, tunable parameter
    t : int
        Iteration index
    opt : int
        Step-size resetting type

    Returns
    -------
    alpha : float
        Resetted step-size. If opt==2 it is greater than the previous one.
    """
    if t == 0:
        return alpha0
    if opt == 0:
        return alpha
    if opt == 1:
        return alpha0
    return alpha * a ** (M / N)


def armijo_condition(x, x_next, X, y, alpha):
    g = 0.5  # gamma
    fun, grad = f_and_df(x, X, y)
    thresh = fun - g * alpha * grad ** 2
    fun_next = logistic(x_next, X, y)
    return fun_next > thresh


# @jit(nopython=True)
def minibatch_gd_armijo(w0, alpha0, M, X, y):
    delta = 0.5
    epochs = 200
    tol = 1e-4
    N, p = X.shape  # number of examples and features
    # weights sequence, w\in\R^p
    w_seq = np.zeros((epochs + 1, p))
    w_seq[0, :] = w0
    # full objective function and full gradient norm sequences
    fun_seq = np.zeros(epochs + 1)
    grad_seq = np.zeros(epochs + 1)
    fun_seq[0], grad_seq[0] = f_and_df(w0, X, y)
    k = 0  # epochs counter   
    while grad_seq[k] > tol * (1 + fun_seq[k]) and k < epochs:
        ## Shuffle dataset
        minibatches = shuffle_dataset(N, k, M)
        ## Approximate gradient and update (internal) weights
        y_seq = np.zeros((len(minibatches) + 1, p)) # internal weights sequence
        y_seq[0, :] = w_seq[k]
        alpha_seq = np.zeros(len(minibatches) + 1)  # step-size per minibatch
        alpha_seq[0] = alpha0
        for t, minibatch in enumerate(minibatches):
            ## Evaluate gradient approximation
            mini_grad = minibatch_gradient(X, y, minibatch, y_seq[t, :], p, M)
            ## Reset step-size
            alpha = reset_step(N, alpha_seq[t], alpha0, M, 5e3, t, 2)
            ## Armijo, compute potential next step
            q = 0  # step-size rejections counter
            y_tnext = y_seq[t, :] - alpha * mini_grad
            while armijo_condition(y_seq[t, :], y_tnext, X, y, alpha):
                alpha = delta * alpha  # reduce step-size
                y_tnext = y_seq[t, :] - alpha * mini_grad
                q += 1
            ## Update (internal) weights
            y_seq[t+1, :] = y_tnext  # internal weights update
            alpha_seq[t+1] = alpha  # accepted step-size
        ## Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_tnext
        fun_seq[k], grad_seq[k] = f_and_df(y_tnext, X, y)
    message = ""
    if grad_seq[-1] <= tol * (1 + np.absolute(fun_seq[-1])):
        message += "Gradient under tolerance"
    if k >= epochs:
        message += "Max epochs exceeded"
    # return w_seq[k,:]
    return w_seq[k, :], grad_seq[k], fun_seq[k], message


#%% Minibatch Gradient Descent with Momentum, fixed step-size and momentum term

# SGDM
def minibatch_gdm_fixed(w0, alpha, beta, M, X, y):
    epochs = 200
    tol = 1e-4
    N, p = X.shape  # number of examples and features
    # weights sequence, w\in\R^p
    w_seq = np.zeros((epochs + 1, p))
    w_seq[0, :] = w0
    # full objective function and full gradient norm sequences
    fun_seq = np.zeros(epochs + 1)
    grad_seq = np.zeros(epochs + 1)
    fun_seq[0], grad_seq[0] = f_and_df(w0, X, y)
    k = 0  # epochs counter
    while grad_seq[k] > tol * (1 + fun_seq[k]) and k < epochs:
        ## Shuffle dataset
        minibatches = shuffle_dataset(N, k, M)
        ## Approximate gradient and update (internal) weights
        # internal weights sequence
        y_seq = np.zeros((len(minibatches) + 1, p))
        y_seq[0, :] = w_seq[k]
        # internal direction sequence, every direction has its own y_t
        d_seq = np.zeros((len(minibatches) + 1, p))
        d_seq[0, :] = np.zeros_like(w_seq[k])
        for t, minibatch in enumerate(minibatches):
            ## Evaluate gradient approximation
            mini_grad = minibatch_gradient(X, y, minibatch, y_seq[t, :], p, M)
            ## Compute direction
            d_seq[t+1, :] = - ((1 - beta) * mini_grad + beta * d_seq[t])
            ## Update (internal) weights
            y_tnext = y_seq[t, :] + alpha * d_seq[t+1, :]
            y_seq[t+1, :] = y_tnext  # internal weights update
        ## Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_tnext
        fun_seq[k], grad_seq[k] = f_and_df(y_tnext, X, y)
    message = ""
    if grad_seq[-1] <= tol * (1 + np.absolute(fun_seq[-1])):
        message += "Gradient under tolerance"
    if k >= epochs:
        message += "Max epochs exceeded"
    # return w_seq[k,:]
    return w_seq[k, :], grad_seq[k], fun_seq[k], message


#%% Minibatch Gradient Descent with Momentum, Armijo line search

# MSL-SGDM-C
def minibatch_gdm_armijo(w0, alpha0, beta, M, X, y):
    delta = 0.5  # consider removing
    epochs = 200  # consider removing
    tol = 1e-4  # consider removing
    N, p = X.shape  # number of examples and features
    # weights sequence, w\in\R^p
    w_seq = np.zeros((epochs + 1, p))
    w_seq[0, :] = w0
    # full objective function and full gradient norm sequences
    fun_seq = np.zeros(epochs + 1)
    grad_seq = np.zeros(epochs + 1)
    fun_seq[0], grad_seq[0] = f_and_df(w0, X, y)
    k = 0  # epochs counter
    while grad_seq[k] > tol * (1 + fun_seq[k]) and k < epochs:
        ## Shuffle dataset
        minibatches = shuffle_dataset(N, k, M)
        # internal weights sequence
        y_seq = np.zeros((len(minibatches) + 1, p))
        y_seq[0, :] = w_seq[k]
        # internal direction sequence, every direction has its own y_t
        d_seq = np.zeros((len(minibatches) + 1, p))
        d_seq[0, :] = np.zeros_like(w_seq[k])
        # internal step-size sequence
        alpha_seq = np.zeros(len(minibatches) + 1)  # step-size per minibatch
        alpha_seq[0] = alpha0
        # internal momentum sequence
        for t, minibatch in enumerate(minibatches):
            ## Evaluate gradient approximation
            mini_grad = minibatch_gradient(X, y, minibatch, y_seq[t, :], p, M)
            ## Compute potential direction
            # d_seq[t+1, :] = - ((1 - beta) * mini_grad + beta * d_seq[t])
            d_tnext =- ((1 - beta) * mini_grad + beta * d_seq[t])
            q_m = 0  # momentum term rejections counter
            while 
            
                beta = delta * beta
            ## Compute potential next step
            q_a = 0  # step-size rejections counter
            y_tnext = y_seq[t, :] - alpha * mini_grad
            while 
            
                alpha = delta * alpha
            ## Update (internal) weights
            y_tnext = y_seq[t, :] + alpha * d_seq[t+1, :]
            y_seq[t+1, :] = y_tnext  # internal weights update
        ## Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_tnext
        fun_seq[k], grad_seq[k] = f_and_df(y_tnext, X, y)
    message = ""
    if grad_seq[-1] <= tol * (1 + np.absolute(fun_seq[-1])):
        message += "Gradient under tolerance"
    if k >= epochs:
        message += "Max epochs exceeded"
    # return w_seq[k,:]
    return w_seq[k, :], grad_seq[k], fun_seq[k], message
