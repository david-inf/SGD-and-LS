# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:15:19 2024

@author: Utente
"""

# %% Packages
import time
import numpy as np
from scipy.optimize import minimize, OptimizeResult
from solvers_utils import logistic, logistic_der, f_and_df, f_and_df_2

# %% Benchmark solver


def l_bfgs_b(w0, X, y):
    res = minimize(f_and_df_2, w0, args=(X, y), method="L-BFGS-B",
                   jac=True, bounds=None, options={"gtol": 1e-4})
    return res

# %% Solvers utils


def minibatch_gradient(X, y, minibatch, x):
    M = minibatch.shape[0]
    samples_x = X[minibatch, :]
    samples_y = y[minibatch]
    grad_sum = logistic_der(x, samples_x, samples_y, M)
    return np.divide(grad_sum, M)


def shuffle_dataset(N, k, M):
    batch = np.arange(N)  # dataset indices, reset every epoch
    rng = np.random.default_rng(k)  # set different seed every epoch
    rng.shuffle(batch)  # shuffle indices
    # array_split is expensive, consider another strategy
    minibatches = np.array_split(batch, N / M)  # create the minibatches
    return minibatches  # list of numpy.ndarray

# %% Minibatch Gradient Descent with fixed step-size

# SGD-Fixed
def minibatch_gd_fixed(w0, alpha, M, X, y):
    epochs = 200
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
    while grad_seq[k] > 1e-3 * (1 + fun_seq[k]) and k < epochs:
        # Shuffle dataset
        minibatches = shuffle_dataset(N, k, M)
        # Approximate gradient and update (internal) weights
        # internal weights sequence
        y_seq = np.zeros((len(minibatches) + 1, p))
        y_seq[0, :] = w_seq[k]
        # for t in range(len(minibatches)):  # for minibatch in minibatches
        for t, minibatch in enumerate(minibatches):
            # Evaluate gradient approximation
            mini_grad = minibatch_gradient(X, y, minibatch, y_seq[t, :])
            # Compute direction
            d_t = - mini_grad
            # Update (internal) weights
            y_seq[t+1, :] = y_seq[t, :] + alpha * d_t  # model internal update
        # Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_seq[-1, :]
        fun_seq[k], grad_seq[k] = f_and_df(y_seq[-1, :], X, y)
    end = time.time()
    message = ""
    if grad_seq[-1] <= 1e-3 * (1 + np.absolute(fun_seq[-1])):
        message += "Gradient under tolerance"
    if k >= epochs:
        message += "Max epochs exceeded"
    return OptimizeResult(fun=fun_seq[k], x=w_seq[k, :], message=message,
                          success=True, solver="MiniGD-fixed", grad=grad_seq[k],
                          fun_per_it=fun_seq, minibatch_size=M,
                          runtime=end - start,
                          step_size=alpha, momentum = 0)

# %% Minibatch Gradient Descent with decreasing step-size

# SGD-Decreasing
def minibatch_gd_decreasing(w0, alpha0, M, X, y):
    epochs = 200
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
    while grad_seq[k] > 1e-3 * (1 + fun_seq[k]) and k < epochs:
        # Shuffle dataset
        minibatches = shuffle_dataset(N, k, M)
        # Approximate gradient and update (internal) weights
        # internal weights sequence
        y_seq = np.zeros((len(minibatches) + 1, p))
        y_seq[0, :] = w_seq[k]
        for t, minibatch in enumerate(minibatches):
            # Evaluate gradient approximation
            mini_grad = minibatch_gradient(X, y, minibatch, y_seq[t, :])
            # Update (internal) weigths
            alpha = alpha0 / (k + 1)
            d_t = - mini_grad
            y_seq[t+1, :] = y_seq[t, :] + alpha * d_t  # model internal update
        # Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_seq[-1, :]
        fun_seq[k], grad_seq[k] = f_and_df(y_seq[-1, :], X, y)
    end = time.time()
    message = ""
    if grad_seq[-1] <= 1e-3 * (1 + np.absolute(fun_seq[-1])):
        message += "Gradient under tolerance"
    if k >= epochs:
        message += "Max epochs exceeded"
    return OptimizeResult(fun=fun_seq[k], x=w_seq[k, :], message=message,
                          success=True, solver="MiniGD-decreasing", grad=grad_seq[k],
                          fun_per_it=fun_seq, minibatch_size=M,
                          runtime=end - start,
                          step_size=alpha0, momentum=0)

# %% Minibatch Gradient Descent with Armijo line search


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
    # N, p = X.shape
    N = X.shape[0]
    alpha = reset_step(N, alpha, alpha0, M, t)
    x_next = x + alpha * d
    q = 0  # step-size rejections counter
    while armijo_condition(x, x_next, X, y, alpha):
        alpha = 0.5 * alpha  # reduce step-size
        x_next = x + alpha * d
        q += 1
    return alpha, x_next


# SGD-Armijo
def minibatch_gd_armijo(w0, alpha0, M, X, y):
    epochs = 200
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
    while grad_seq[k] > 1e-3 * (1 + fun_seq[k]) and k < epochs:
        # Shuffle dataset
        minibatches = shuffle_dataset(N, k, M)
        # Approximate gradient and update (internal) weights
        # internal weights sequence
        y_seq = np.zeros((len(minibatches) + 1, p))
        y_seq[0, :] = w_seq[k]
        alpha_seq = np.zeros(len(minibatches) + 1)  # step-size per minibatch
        alpha_seq[0] = alpha0
        for t, minibatch in enumerate(minibatches):
            # Evaluate gradient approximation
            mini_grad = minibatch_gradient(X, y, minibatch, y_seq[t, :])
            d_t = - mini_grad
            # Armijo line search
            alpha_seq[t+1], y_seq[t+1, :] = armijo_method(
                y_seq[t, :], d_t, X, y, alpha_seq[t], alpha0, M, t)
        # Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_seq[-1, :]
        fun_seq[k], grad_seq[k] = f_and_df(y_seq[-1, :], X, y)
    end = time.time()
    message = ""
    if grad_seq[-1] <= 1e-3 * (1 + np.absolute(fun_seq[-1])):
        message += "Gradient under tolerance"
    if k >= epochs:
        message += "Max epochs exceeded"
    return OptimizeResult(fun=fun_seq[k], x=w_seq[k, :], message=message,
                          success=True, solver="MiniGD-Armijo", grad=grad_seq[k],
                          fun_per_it=fun_seq, minibatch_size=M,
                          runtime=end - start,
                          step_size=alpha0, momentum=0)

# %% Minibatch Gradient Descent with Momentum, fixed step-size and momentum term

# SGDM
def minibatch_gdm_fixed(w0, alpha, beta, M, X, y):
    epochs = 200
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
    while grad_seq[k] > 1e-3 * (1 + fun_seq[k]) and k < epochs:
        # Shuffle dataset
        minibatches = shuffle_dataset(N, k, M)
        # Approximate gradient and update (internal) weights
        # internal weights sequence
        y_seq = np.zeros((len(minibatches) + 1, p))
        y_seq[0, :] = w_seq[k]
        # internal direction sequence, every direction has its own y_t
        d_seq = np.zeros((len(minibatches) + 1, p))
        d_seq[0, :] = np.zeros_like(w_seq[k])
        for t, minibatch in enumerate(minibatches):
            # Evaluate gradient approximation
            mini_grad = minibatch_gradient(X, y, minibatch, y_seq[t, :])
            # Compute direction
            d_seq[t+1, :] = - ((1 - beta) * mini_grad + beta * d_seq[t])
            # Update (internal) weights
            y_seq[t+1, :] = y_seq[t, :] + alpha * d_seq[t+1, :]
        # Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_seq[-1, :]
        fun_seq[k], grad_seq[k] = f_and_df(y_seq[-1, :], X, y)
    end = time.time()
    message = ""
    if grad_seq[-1] <= 1e-3 * (1 + np.absolute(fun_seq[-1])):
        message += "Gradient under tolerance"
    if k >= epochs:
        message += "Max epochs exceeded"
    return OptimizeResult(fun=fun_seq[k], x=w_seq[k, :], message=message,
                          success=True, solver="MiniGDM-fixed", grad=grad_seq[k],
                          fun_per_it=fun_seq, minibatch_size=M,
                          runtime=end - start,
                          step_size=alpha, momentum=beta)

# %% Minibatch Gradient Descent with Momentum, Armijo line search


def direction_condition(grad, d):
    # check grad and direction dimension
    return np.dot(grad, d) < 0  # d non-descent


def momentum_correction(beta0, d, grad):
    beta = beta0
    # compute potential next direction
    d_next = - ((1 - beta) * grad + beta * d)
    q = 0  # momentum term rejections counter
    while not direction_condition(grad, d_next):
        beta = 0.5 * beta  # reduce momentum term
        d_next = - ((1 - beta) * grad + beta * d)
        q += 1
    return beta, d_next


# MSL-SGDM-C
def msl_sgdm_c(w0, alpha0, beta0, M, X, y):
    epochs = 200  # consider removing
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
    while grad_seq[k] > 1e-3 * (1 + fun_seq[k]) and k < epochs:
        # Shuffle dataset
        minibatches = shuffle_dataset(N, k, M)
        # internal weights sequence
        y_seq = np.zeros((len(minibatches) + 1, p))
        y_seq[0, :] = w_seq[k]
        # internal direction sequence, every direction has its own y_t
        d_seq = np.zeros((len(minibatches) + 1, p))
        d_seq[0, :] = np.zeros_like(w_seq[k])  # d_0 = 0
        # internal step-size sequence
        alpha_seq = np.zeros(len(minibatches) + 1)  # step-size per minibatch
        alpha_seq[0] = alpha0
        # internal momentum sequence
        beta_seq = np.zeros(len(minibatches) + 1)
        beta_seq[0] = beta0
        for t, minibatch in enumerate(minibatches):
            # Evaluate gradient approximation
            mini_grad = minibatch_gradient(X, y, minibatch, y_seq[t, :])
            # Momentum correction
            beta_seq[t+1], d_seq[t+1, :] = momentum_correction(
                beta0, d_seq[t], mini_grad)
            # Armijo line search
            alpha_seq[t+1], y_seq[t+1, :] = armijo_method(
                y_seq[t, :], d_seq[t+1, :], X, y, alpha_seq[t], alpha0, M, t)
        # Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_seq[-1, :]
        fun_seq[k], grad_seq[k] = f_and_df(y_seq[-1], X, y)
    end = time.time()
    message = ""
    if grad_seq[-1] <= 1e-3 * (1 + np.absolute(fun_seq[-1])):
        message += "Gradient under tolerance"
    if k >= epochs:
        message += "Max epochs exceeded"
    return OptimizeResult(fun=fun_seq[k], x=w_seq[k, :], message=message,
                          success=True, solver="MSL-SGDM-C", grad=grad_seq[k],
                          fun_per_it=fun_seq, minibatch_size=M,
                          runtime=end - start,
                          step_size=alpha0, momentum=beta0)


# MSL-SGDM-R
def msl_sgdm_r(w0, alpha0, beta0, M, X, y):
    epochs = 200  # consider removing
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
    while grad_seq[k] > 1e-3 * (1 + fun_seq[k]) and k < epochs:
        # Shuffle dataset
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
        for t, minibatch in enumerate(minibatches):
            # Evaluate gradient approximation
            mini_grad = minibatch_gradient(X, y, minibatch, y_seq[t, :])
            # Compute potential direction
            d_tnext = - ((1 - beta0) * mini_grad + beta0 * d_seq[t])
            if not direction_condition(mini_grad, d_tnext):
                d_tnext = d_seq[0, :]
            d_seq[t+1, :] = d_tnext
            # Armijo line search
            alpha_seq[t+1], y_seq[t+1, :] = armijo_method(
                y_seq[t, :], d_tnext, X, y, alpha_seq[t], alpha0, M, t)
        # Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_seq[-1, :]
        fun_seq[k], grad_seq[k] = f_and_df(y_seq[-1, :], X, y)
    end = time.time()
    message = ""
    if grad_seq[-1] <= 1e-3 * (1 + np.absolute(fun_seq[-1])):
        message += "Gradient under tolerance"
    if k >= epochs:
        message += "Max epochs exceeded"
    return OptimizeResult(fun=fun_seq[k], x=w_seq[k, :], message=message,
                          success=True, solver="MSL-SGDM-R", grad=grad_seq[k],
                          fun_per_it=fun_seq, minibatch_size=M,
                          runtime=end - start,
                          step_size=alpha0, momentum=beta0)
