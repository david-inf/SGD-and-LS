# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:15:19 2024

@author: Utente
"""

#%% Packages
import time
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult
# from numba import jit
from solvers_utils import logistic, logistic_der, f_and_df, f_and_df_2

#%% Benchmark solver
def l_bfgs_b(w0, X, y):
    res = minimize(f_and_df_2, w0, args=(X, y), method="L-BFGS-B",
                   jac=True, bounds=None, options={"gtol": 1e-4})
    return res    

#%% Solvers utils
def minibatch_gradient(X, y, minibatch, y_tbefore):
    p = X.shape[1]
    M = minibatch.shape[0]
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
    # array_split is expensive, consider another strategy
    minibatches = np.array_split(batch, N / M)  # create the minibatches
    return minibatches  # list of numpy.ndarray

#%% Minibatch Gradient Descent with fixed step-size
def minibatch_gd_fixed(w0, alpha, M, X, y):
    epochs = 200
    # tol = 1e-4
    N, p = X.shape  # number of examples and features
    # weights sequence, w\in\R^p
    w_seq = np.zeros((epochs + 1, p))
    w_seq[0, :] = w0
    # full objective function and full gradient norm sequences
    fun_seq = np.zeros(epochs + 1)
    grad_seq = np.zeros(epochs + 1)
    fun_seq[0], grad_seq[0] = f_and_df(w0, X, y)
    start = time.time()
    k = 0  # epochs counter
    while grad_seq[k] > 1e-4 * (1 + fun_seq[k]) and k < epochs:
        ## Shuffle dataset
        minibatches = shuffle_dataset(N, k, M)
        ## Approximate gradient and update (internal) weights
        # internal weights sequence
        y_seq = np.zeros((len(minibatches) + 1, p))
        y_seq[0, :] = w_seq[k]
        # for t in range(len(minibatches)):  # for minibatch in minibatches
        for t, minibatch in enumerate(minibatches):
            ## Evaluate gradient approximation
            mini_grad = minibatch_gradient(X, y, minibatch, y_seq[t, :])
            ## Compute direction
            # d_t = - mini_grad
            ## Update (internal) weights
            y_tnext = y_seq[t, :] - alpha * mini_grad  # model internal update
            y_seq[t+1, :] = y_tnext  # internal weights update
        ## Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_tnext
        fun_seq[k], grad_seq[k] = f_and_df(y_tnext, X, y)
    end = time.time()
    message = ""
    if grad_seq[-1] <= 1e-4 * (1 + np.absolute(fun_seq[-1])):
        message += "Gradient under tolerance"
    if k >= epochs:
        message += "Max epochs exceeded"
    # return w_seq[k,:]
    # return w_seq[k, :], grad_seq[k], fun_seq[k], message
    return OptimizeResult(fun=fun_seq[k], x=w_seq[k,:], message=message,
                success=True, solver="MiniGD-fixed", grad=grad_seq[k],
                fun_per_it=fun_seq, minibatch_size=M,
                runtime = end - start, step_size=alpha)

#%% Minibatch Gradient Descent with decreasing step-size
def minibatch_gd_decreasing(w0, alpha0, M, X, y):
    epochs = 200
    # tol = 1e-4
    N, p = X.shape  # number of examples and features
    # weights sequence, w\in\R^p
    w_seq = np.zeros((epochs + 1, p))
    w_seq[0, :] = w0
    # full objective function and full gradient norm sequences
    fun_seq = np.zeros(epochs + 1)
    grad_seq = np.zeros(epochs + 1)
    fun_seq[0], grad_seq[0] = f_and_df(w0, X, y)
    start = time.time()
    k = 0  # epochs counter
    while grad_seq[k] > 1e-4 * (1 + fun_seq[k]) and k < epochs:
        ## Shuffle dataset
        minibatches = shuffle_dataset(N, k, M)
        ## Approximate gradient and update (internal) weights
        # internal weights sequence
        y_seq = np.zeros((len(minibatches) + 1, p))
        y_seq[0, :] = w_seq[k]
        for t, minibatch in enumerate(minibatches):
            ## Evaluate gradient approximation
            mini_grad = minibatch_gradient(X, y, minibatch, y_seq[t, :])
            ## Update (internal) weigths
            alpha = alpha0 / (k + 1)
            y_tnext = y_seq[t, :] - alpha * mini_grad  # model internal update
            y_seq[t+1, :] = y_tnext  # internal weights update
        ## Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_tnext
        fun_seq[k], grad_seq[k] = f_and_df(y_tnext, X, y)
    end = time.time()
    message = ""
    if grad_seq[-1] <= 1e-4 * (1 + np.absolute(fun_seq[-1])):
        message += "Gradient under tolerance"
    if k >= epochs:
        message += "Max epochs exceeded"
    # return w_seq[k,:]
    # return w_seq[k, :], grad_seq[k], fun_seq[k], message
    return OptimizeResult(fun=fun_seq[k], x=w_seq[k,:], message=message,
                success=True, solver="MiniGD-decreasing", grad=grad_seq[k],
                fun_per_it=fun_seq, minibatch_size=M,
                runtime = end - start, initial_step_size=alpha0)

#%% Minibatch Gradient Descent with Armijo line search
def reset_step(N, alpha, alpha0, M, t):
    opt = 2
    a = 5e2
    if t == 0:
        return alpha0
    if opt == 0:
        return alpha
    if opt == 1:
        return alpha0
    return alpha * a ** (M / N)


def armijo_condition(x, x_next, X, y, alpha):
    g = 0.5  # gamma
    fun, grad_norm = f_and_df(x, X, y)
    thresh = fun - g * alpha * grad_norm ** 2
    fun_next = logistic(x_next, X, y)
    return fun_next > thresh


def armijo_method(x, d, X, y, alpha, alpha0, M, t):
    N, p = X.shape
    alpha = reset_step(N, alpha, alpha0, M, t)
    x_next = x + alpha * d
    q = 0  # step-size rejections counter
    while armijo_condition(x, x_next, X, y, alpha):
        alpha = 0.5 * alpha  # reduce step-size
        x_next = x + alpha * d
        q += 1
    return alpha, x_next


# @jit(nopython=True)
def minibatch_gd_armijo(w0, alpha0, M, X, y):
    # delta = 0.5
    epochs = 200
    # tol = 1e-4
    N, p = X.shape  # number of examples and features
    # weights sequence, w\in\R^p
    w_seq = np.zeros((epochs + 1, p))
    w_seq[0, :] = w0
    # full objective function and full gradient norm sequences
    fun_seq = np.zeros(epochs + 1)
    grad_seq = np.zeros(epochs + 1)
    fun_seq[0], grad_seq[0] = f_and_df(w0, X, y)
    start = time.time()
    k = 0  # epochs counter   
    while grad_seq[k] > 1e-4 * (1 + fun_seq[k]) and k < epochs:
        ## Shuffle dataset
        minibatches = shuffle_dataset(N, k, M)
        ## Approximate gradient and update (internal) weights
        y_seq = np.zeros((len(minibatches) + 1, p)) # internal weights sequence
        y_seq[0, :] = w_seq[k]
        alpha_seq = np.zeros(len(minibatches) + 1)  # step-size per minibatch
        alpha_seq[0] = alpha0
        for t, minibatch in enumerate(minibatches):
            ## Evaluate gradient approximation
            mini_grad = minibatch_gradient(X, y, minibatch, y_seq[t, :])
            ## Armijo line search
            alpha_seq[t+1], y_tnext = armijo_method(
                y_seq[t, :], -mini_grad, X, y, alpha_seq[t], alpha0, M, t)
            y_seq[t+1, :] = y_tnext  # internal weights update
        ## Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_tnext
        fun_seq[k], grad_seq[k] = f_and_df(y_tnext, X, y)
    end = time.time()
    message = ""
    if grad_seq[-1] <= 1e-4 * (1 + np.absolute(fun_seq[-1])):
        message += "Gradient under tolerance"
    if k >= epochs:
        message += "Max epochs exceeded"
    # return w_seq[k,:]
    # return w_seq[k, :], grad_seq[k], fun_seq[k], message
    return OptimizeResult(fun=fun_seq[k], x=w_seq[k,:], message=message,
                success=True, solver="MiniGD-Armijo", grad=grad_seq[k],
                fun_per_it=fun_seq, minibatch_size=M,
                runtime = end - start)

#%% Minibatch Gradient Descent with Momentum, fixed step-size and momentum term

# SGDM
def minibatch_gdm_fixed(w0, alpha, beta, M, X, y):
    epochs = 200
    # tol = 1e-4
    N, p = X.shape  # number of examples and features
    # weights sequence, w\in\R^p
    w_seq = np.zeros((epochs + 1, p))
    w_seq[0, :] = w0
    # full objective function and full gradient norm sequences
    fun_seq = np.zeros(epochs + 1)
    grad_seq = np.zeros(epochs + 1)
    fun_seq[0], grad_seq[0] = f_and_df(w0, X, y)
    start = time.time()
    k = 0  # epochs counter
    while grad_seq[k] > 1e-4 * (1 + fun_seq[k]) and k < epochs:
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
            mini_grad = minibatch_gradient(X, y, minibatch, y_seq[t, :])
            ## Compute direction
            d_seq[t+1, :] = - ((1 - beta) * mini_grad + beta * d_seq[t])
            ## Update (internal) weights
            y_tnext = y_seq[t, :] + alpha * d_seq[t+1, :]
            y_seq[t+1, :] = y_tnext  # internal weights update
        ## Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_tnext
        fun_seq[k], grad_seq[k] = f_and_df(y_tnext, X, y)
    end = time.time()
    message = ""
    if grad_seq[-1] <= 1e-4 * (1 + np.absolute(fun_seq[-1])):
        message += "Gradient under tolerance"
    if k >= epochs:
        message += "Max epochs exceeded"
    # return w_seq[k,:]
    # return w_seq[k, :], grad_seq[k], fun_seq[k], message
    return OptimizeResult(fun=fun_seq[k], x=w_seq[k,:], message=message,
                success=True, solver="MiniGDM-fixed", grad=grad_seq[k],
                fun_per_it=fun_seq, minibatch_size=M,
                runtime = end - start, step_size=alpha, momentum=beta)

#%% Minibatch Gradient Descent with Momentum, Armijo line search

def direction_condition(grad, d):
    # check grad and direction dimension
    return np.dot(grad, d) < 0  # d non-descent


# MSL-SGDM-C
def msl_sgdm_c(w0, alpha0, beta0, M, X, y):
    # delta = 0.5  # consider removing
    epochs = 200  # consider removing
    # tol = 1e-4  # consider removing
    N, p = X.shape  # number of examples and features
    # weights sequence, w\in\R^p
    w_seq = np.zeros((epochs + 1, p))
    w_seq[0, :] = w0
    # full objective function and full gradient norm sequences
    fun_seq = np.zeros(epochs + 1)
    grad_seq = np.zeros(epochs + 1)
    fun_seq[0], grad_seq[0] = f_and_df(w0, X, y)
    start = time.time()
    k = 0  # epochs counter
    while grad_seq[k] > 1e-4 * (1 + fun_seq[k]) and k < epochs:
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
        # beta_seq = np.zeros(len(minibatches) + 1)
        for t, minibatch in enumerate(minibatches):
            ## Evaluate gradient approximation
            mini_grad = minibatch_gradient(X, y, minibatch, y_seq[t, :])
            ## Compute potential direction
            beta = beta0
            # beta = reset_momentum()
            d_tnext = - ((1 - beta) * mini_grad + beta * d_seq[t])
            q_m = 0  # momentum term rejections counter
            while not direction_condition(mini_grad, d_tnext):
                beta = 0.5 * beta  # reduce momentum term
                d_tnext = - ((1 - beta) * mini_grad + beta * d_seq[t])
                q_m += 1
            d_seq[t+1,:] = d_tnext
            ## Compute potential next step
            alpha = reset_step(N, alpha_seq[t], alpha0, M, t)
            y_tnext = y_seq[t, :] + alpha * d_tnext
            q_a = 0  # step-size rejections counter
            while armijo_condition(y_seq[t, :], y_tnext, X, y, alpha):
                alpha = 0.5 * alpha  # reduce step-size
                y_tnext = y_seq[t, :] - alpha * mini_grad
                q_a += 1
            alpha_seq[t+1] = alpha  # accepted step-size
            ## Update (internal) weights
            y_seq[t+1, :] = y_tnext  # internal weights update
        ## Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_tnext
        fun_seq[k], grad_seq[k] = f_and_df(y_tnext, X, y)
    end = time.time()
    message = ""
    if grad_seq[-1] <= 1e-4 * (1 + np.absolute(fun_seq[-1])):
        message += "Gradient under tolerance"
    if k >= epochs:
        message += "Max epochs exceeded"
    # return w_seq[k,:]
    # return w_seq[k, :], grad_seq[k], fun_seq[k], message
    return OptimizeResult(fun=fun_seq[k], x=w_seq[k,:], message=message,
                success=True, solver="MSL-SGDM-C", grad=grad_seq[k],
                fun_per_it=fun_seq, minibatch_size=M,
                runtime = end - start)


# MSL-SGDM-R
def msl_sgdm_r(w0, alpha0, beta0, M, X, y):
    # delta = 0.5  # consider removing
    epochs = 200  # consider removing
    # tol = 1e-4  # consider removing
    N, p = X.shape  # number of examples and features
    # weights sequence, w\in\R^p
    w_seq = np.zeros((epochs + 1, p))
    w_seq[0, :] = w0
    # full objective function and full gradient norm sequences
    fun_seq = np.zeros(epochs + 1)
    grad_seq = np.zeros(epochs + 1)
    fun_seq[0], grad_seq[0] = f_and_df(w0, X, y)
    start = time.time()
    k = 0  # epochs counter
    while grad_seq[k] > 1e-4 * (1 + fun_seq[k]) and k < epochs:
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
        # beta_seq = np.zeros(len(minibatches) + 1)
        for t, minibatch in enumerate(minibatches):
            ## Evaluate gradient approximation
            mini_grad = minibatch_gradient(X, y, minibatch, y_seq[t, :])
            ## Compute potential direction
            d_tnext = - ((1 - beta0) * mini_grad + beta0 * d_seq[t])
            if not direction_condition(mini_grad, d_tnext):
                d_tnext = d_seq[0,:]
            d_seq[t+1,:] = d_tnext
            ## Compute potential next step
            alpha = reset_step(N, alpha_seq[t], alpha0, M, t)
            y_tnext = y_seq[t, :] + alpha * d_tnext
            q_a = 0  # step-size rejections counter
            while armijo_condition(y_seq[t, :], y_tnext, X, y, alpha):
                alpha = 0.5 * alpha  # reduce step-size
                y_tnext = y_seq[t, :] - alpha * mini_grad
                q_a += 1
            ## Update (internal) weights
            alpha_seq[t+1] = alpha  # accepted step-size
            y_seq[t+1, :] = y_tnext  # internal weights update
        ## Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_tnext
        fun_seq[k], grad_seq[k] = f_and_df(y_tnext, X, y)
    end = time.time()
    message = ""
    if grad_seq[-1] <= 1e-4 * (1 + np.absolute(fun_seq[-1])):
        message += "Gradient under tolerance"
    if k >= epochs:
        message += "Max epochs exceeded"
    # return w_seq[k,:]
    # return w_seq[k, :], grad_seq[k], fun_seq[k], message
    return OptimizeResult(fun=fun_seq[k], x=w_seq[k,:], message=message,
                success=True, solver="MSL-SGDM-R", grad=grad_seq[k],
                fun_per_it=fun_seq, minibatch_size=M,
                runtime = end - start)
