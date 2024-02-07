# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:15:19 2024

@author: Utente
"""

# %% Packages
import time
import numpy as np
from scipy.optimize import minimize, OptimizeResult
from solvers_utils import logistic, logistic_der, f_and_dfnorm, f_and_df

# %% [0] L-BFGS-B


def l_bfgs_b(w0, X, y, lam):
    res = minimize(f_and_df, w0, args=(X, y, lam), method="L-BFGS-B",
                   jac=True, bounds=None, options={"gtol": 1e-5, "ftol": 1e-5})
    return res

# %% Solvers utils


def minibatch_gradient(x, X, y, lam, minibatch):
    # x: evaluate gradient at this value
    M = minibatch.shape[0]
    samples_x = X[minibatch, :]
    samples_y = y[minibatch]
    # in oder to sum over the gradients, regularization term is multiplied by M
    grad_sum = logistic_der(x, samples_x, samples_y, lam, M)
    return grad_sum / M


def shuffle_dataset(N, k, M):
    batch = np.arange(N)  # dataset indices, reset every epoch
    rng = np.random.default_rng(k)  # set different seed every epoch
    rng.shuffle(batch)  # shuffle indices
    # array_split is expensive, consider another strategy
    minibatches = np.array_split(batch, N / M)  # create the minibatches
    return minibatches  # list of numpy.ndarray


def stopping(fun_k, grad_k, nit, max_iter, tol):
    # fun and grad already evaluated
    # grad > 1e-3 * (1 + fun) and k < epochs:
    return grad_k > tol and nit < max_iter

# %% [1] SGD-Fixed

# minimize(logistic, w0, args=(),method=sgd_fixed,
#       jac=logistic_der, options=dict(X, y, lam, M=16, alpha=1))

# Minibatch Gradient Descent with fixed step-size


def sgd_fixed(w0, X, y, lam, M, alpha, epochs, tol):
    N, p = X.shape
    # weights sequence
    w_seq = np.zeros((epochs + 1, p))
    w_seq[0, :] = w0
    # full objective function and full gradient norm sequences
    fun_seq = np.zeros(epochs + 1)
    grad_seq = np.zeros(epochs + 1)
    fun_seq[0], grad_seq[0] = f_and_dfnorm(w0, X, y, lam)
    # function and gradient evaluations counters
    # funcalls = 1
    # gradcalls = 1
    start = time.time()
    k = 0  # epochs counter
    while stopping(fun_seq[k], grad_seq[k], k, epochs, tol):
        minibatches = shuffle_dataset(N, k, M)  # get random minibatches
        # internal weights sequence
        y_seq = np.zeros((len(minibatches) + 1, p))
        y_seq[0, :] = w_seq[k]
        for t, minibatch in enumerate(minibatches):
            # Evaluate gradient approximation
            mini_grad = minibatch_gradient(y_seq[t, :], X, y, lam, minibatch)
            d_t = - mini_grad  # compute direction
            y_seq[t+1, :] = y_seq[t, :] + alpha * d_t  # model internal update
        # Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_seq[-1, :]
        fun_seq[k], grad_seq[k] = f_and_dfnorm(y_seq[-1, :], X, y, lam)
    end = time.time()
    return OptimizeResult(fun=fun_seq[k], x=w_seq[k, :],
                          success=(k > 1), solver="SGD-Fixed",
                          grad=grad_seq[k], fun_per_it=fun_seq,
                          minibatch_size=M, nit=k,
                          runtime=(end - start),
                          step_size=alpha, momentum=0)

# %% [2] SGD-Decreasing


# Minibatch Gradient Descent with decreasing step-size
def sgd_decreasing(w0, X, y, lam, M, alpha0, epochs, tol):
    N, p = X.shape  # number of examples and features
    # weights sequence
    w_seq = np.zeros((epochs + 1, p))
    w_seq[0, :] = w0
    # full objective function and full gradient norm sequences
    fun_seq = np.zeros(epochs + 1)
    grad_seq = np.zeros(epochs + 1)
    fun_seq[0], grad_seq[0] = f_and_dfnorm(w0, X, y, lam)
    start = time.time()
    k = 0  # epochs counter
    while stopping(fun_seq[k], grad_seq[k], k, epochs, tol):
        minibatches = shuffle_dataset(N, k, M)  # get random minibatches
        # internal weights sequence
        y_seq = np.zeros((len(minibatches) + 1, p))
        y_seq[0, :] = w_seq[k]
        for t, minibatch in enumerate(minibatches):
            mini_grad = minibatch_gradient(y_seq[t, :], X, y, lam, minibatch)
            d_t = - mini_grad
            alpha = alpha0 / (k + 1)
            y_seq[t+1, :] = y_seq[t, :] + alpha * d_t  # model internal update
        # Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_seq[-1, :]
        fun_seq[k], grad_seq[k] = f_and_dfnorm(y_seq[-1, :], X, y, lam)
    end = time.time()
    return OptimizeResult(fun=fun_seq[k], x=w_seq[k, :],
                          success=(k > 1), solver="SGD-Decreasing",
                          grad=grad_seq[k], fun_per_it=fun_seq,
                          minibatch_size=M, nit=k,
                          runtime=(end - start),
                          step_size=alpha0, momentum=0)

# %% [3] SGD-Armijo


def reset_step(N, alpha, alpha0, M, t):
    # alpha: previous iteration step-size
    opt = 2
    a = 5e2
    if t == 0:
        return alpha0
    if opt == 0:
        return alpha
    if opt == 1:
        return alpha0
    return alpha * a ** (M / N)


def armijo_condition(x, x_next, X, y, lam, alpha):
    g = 0.5  # gamma
    fun, grad_norm = f_and_dfnorm(x, X, y, lam)
    thresh = fun - g * alpha * grad_norm ** 2
    fun_next = logistic(x_next, X, y)
    return fun_next > thresh


def armijo_method(x, d, X, y, lam, alpha, alpha0, M, t):
    # N, p = X.shape
    N = X.shape[0]
    alpha = reset_step(N, alpha, alpha0, M, t)
    x_next = x + alpha * d
    q = 0  # step-size rejections counter
    while armijo_condition(x, x_next, X, y, lam, alpha) and q < 20:
        alpha = 0.5 * alpha  # reduce step-size
        x_next = x + alpha * d
        q += 1
    return alpha, x_next


# Minibatch Gradient Descent with Armijo line search
def sgd_armijo(w0, X, y, lam, M, alpha0, epochs, tol):
    N, p = X.shape  # number of examples and features
    # weights sequence, w\in\R^p
    w_seq = np.zeros((epochs + 1, p))
    w_seq[0, :] = w0
    # full objective function and full gradient norm sequences
    fun_seq = np.zeros(epochs + 1)
    grad_seq = np.zeros(epochs + 1)
    fun_seq[0], grad_seq[0] = f_and_dfnorm(w0, X, y, lam)
    start = time.time()
    k = 0  # epochs counter
    while stopping(fun_seq[k], grad_seq[k], k, epochs, tol):
        minibatches = shuffle_dataset(N, k, M)  # get random minibatches
        # internal weights sequence
        y_seq = np.zeros((len(minibatches) + 1, p))
        y_seq[0, :] = w_seq[k]
        # step-size for every minibatch
        alpha_seq = np.zeros(len(minibatches) + 1)  # step-size per minibatch
        alpha_seq[0] = alpha0
        for t, minibatch in enumerate(minibatches):
            mini_grad = minibatch_gradient(y_seq[t, :], X, y, lam, minibatch)
            d_t = - mini_grad  # compute direction
            # Armijo line search
            alpha_seq[t+1], y_seq[t+1, :] = armijo_method(
                y_seq[t, :], d_t, X, y, lam, alpha_seq[t], alpha0, M, t)
        # Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_seq[-1, :]
        fun_seq[k], grad_seq[k] = f_and_dfnorm(y_seq[-1, :], X, y, lam)
    end = time.time()
    return OptimizeResult(fun=fun_seq[k], x=w_seq[k, :],
                          success=(k > 1), solver="SGD-Armijo",
                          grad=grad_seq[k], fun_per_it=fun_seq,
                          minibatch_size=M, nit=k,
                          runtime=(end - start),
                          step_size=alpha0, momentum=0)

# %% [4] SGDM


# Minibatch Gradient Descent with Momentum, fixed step-size and momentum term
def sgdm(w0, X, y, lam, M, alpha, beta, epochs, tol):
    N, p = X.shape  # number of examples and features
    # weights sequence
    w_seq = np.zeros((epochs + 1, p))
    w_seq[0, :] = w0
    # full objective function and full gradient norm sequences
    fun_seq = np.zeros(epochs + 1)
    grad_seq = np.zeros(epochs + 1)
    fun_seq[0], grad_seq[0] = f_and_dfnorm(w0, X, y, lam)
    start = time.time()
    k = 0  # epochs counter
    while stopping(fun_seq[k], grad_seq[k], k, epochs, tol):
        minibatches = shuffle_dataset(N, k, M)  # get random minibatches
        # internal weights sequence
        y_seq = np.zeros((len(minibatches) + 1, p))
        y_seq[0, :] = w_seq[k]
        # internal direction sequence, every direction has its own y_t
        d_seq = np.zeros((len(minibatches) + 1, p))
        d_seq[0, :] = np.zeros_like(w_seq[k])
        for t, minibatch in enumerate(minibatches):
            mini_grad = minibatch_gradient(y_seq[t, :], X, y, lam, minibatch)
            # Compute direction
            d_seq[t+1, :] = - ((1 - beta) * mini_grad + beta * d_seq[t])
            y_seq[t+1, :] = y_seq[t, :] + alpha * d_seq[t+1, :]
        # Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_seq[-1, :]
        fun_seq[k], grad_seq[k] = f_and_dfnorm(y_seq[-1, :], X, y, lam)
    end = time.time()
    return OptimizeResult(fun=fun_seq[k], x=w_seq[k, :],
                          success=(k > 1), solver="SGDM",
                          grad=grad_seq[k], fun_per_it=fun_seq,
                          minibatch_size=M, nit=k,
                          runtime=(end - start),
                          step_size=alpha, momentum=beta)

# %% [5] MSL-SGDM-C/R


def direction_condition(grad, d):
    # check grad and direction dimension
    return np.dot(grad, d) < 0  # d non-descent


def momentum_correction(beta0, d, grad):
    beta = beta0
    # compute potential next direction
    d_next = - ((1 - beta) * grad + beta * d)
    q = 0  # momentum term rejections counter
    while not direction_condition(grad, d_next) and q < 20:
        beta = 0.5 * beta  # reduce momentum term
        d_next = - ((1 - beta) * grad + beta * d)
        q += 1
    return beta, d_next


# Minibatch Gradient Descent with Momentum correction, Armijo line search
def msl_sgdm_c(w0, X, y, lam, M, alpha0, beta0, epochs, tol):
    N, p = X.shape  # number of examples and features
    # weights sequence
    w_seq = np.zeros((epochs + 1, p))
    w_seq[0, :] = w0
    # full objective function and full gradient norm sequences
    fun_seq = np.zeros(epochs + 1)
    grad_seq = np.zeros(epochs + 1)
    fun_seq[0], grad_seq[0] = f_and_dfnorm(w0, X, y, lam)
    start = time.time()
    k = 0  # epochs counter
    while stopping(fun_seq[k], grad_seq[k], k, epochs, tol):
        minibatches = shuffle_dataset(N, k, M)  # get random minibatches
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
            mini_grad = minibatch_gradient(y_seq[t, :], X, y, lam, minibatch)
            # Momentum correction
            beta_seq[t+1], d_seq[t+1, :] = momentum_correction(
                beta0, d_seq[t], mini_grad)
            # Armijo line search
            alpha_seq[t+1], y_seq[t+1, :] = armijo_method(
                y_seq[t, :], d_seq[t+1, :], X, y, lam, alpha_seq[t], alpha0, M, t)
        # Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_seq[-1, :]
        fun_seq[k], grad_seq[k] = f_and_dfnorm(y_seq[-1], X, y, lam)
    end = time.time()
    return OptimizeResult(fun=fun_seq[k], x=w_seq[k, :],
                          success=(k > 1), solver="SGDM",
                          grad=grad_seq[k], fun_per_it=fun_seq,
                          minibatch_size=M, nit=k,
                          runtime=(end - start),
                          step_size=alpha0, momentum=beta0)


# Minibatch Gradient Descent with Momentum restart, Armijo line search
def msl_sgdm_r(w0, X, y, lam, M, alpha0, beta0, epochs, tol):
    N, p = X.shape  # number of examples and features
    # weights sequence
    w_seq = np.zeros((epochs + 1, p))
    w_seq[0, :] = w0
    # full objective function and full gradient norm sequences
    fun_seq = np.zeros(epochs + 1)
    grad_seq = np.zeros(epochs + 1)
    fun_seq[0], grad_seq[0] = f_and_dfnorm(w0, X, y, lam)
    start = time.time()
    k = 0  # epochs counter
    while stopping(fun_seq[k], grad_seq[k], k, epochs, tol):
        minibatches = shuffle_dataset(N, k, M)  # get random minibatches
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
            mini_grad = minibatch_gradient(y_seq[t, :], X, y, lam, minibatch)
            # Compute potential direction
            d_tnext = - ((1 - beta0) * mini_grad + beta0 * d_seq[t])
            if not direction_condition(mini_grad, d_tnext):
                d_tnext = d_seq[0, :]
            d_seq[t+1, :] = d_tnext
            # Armijo line search
            alpha_seq[t+1], y_seq[t+1, :] = armijo_method(
                y_seq[t, :], d_tnext, X, y, lam, alpha_seq[t], alpha0, M, t)
        # Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_seq[-1, :]
        fun_seq[k], grad_seq[k] = f_and_dfnorm(y_seq[-1, :], X, y, lam)
    end = time.time()
    return OptimizeResult(fun=fun_seq[k], x=w_seq[k, :],
                          success=(k > 1), solver="SGDM",
                          grad=grad_seq[k], fun_per_it=fun_seq,
                          minibatch_size=M, nit=k,
                          runtime=(end - start),
                          step_size=alpha0, momentum=beta0)
