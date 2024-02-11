# -*- coding: utf-8 -*-

# %% Packages
import time
import numpy as np
from scipy.optimize import minimize, OptimizeResult
from solvers_utils import logistic, logistic_der, f_and_df, logistic_hess

# %% [0] L-BFGS-B / Newton-CG / CG


def l_bfgs_b(w0, X, y, lam):
    res = minimize(f_and_df, w0, args=(X, y, lam), method="L-BFGS-B",
                   jac=True, bounds=None,
                   options={"gtol": 1e-5, "ftol": 1e-5})
    return res


def newton_cg(w0, X, y, lam):
    res = minimize(f_and_df, w0, args=(X, y, lam), method="Newton-CG",
                   jac=True, hess=logistic_hess, bounds=None)
    return res


def cg(w0, X, y, lam):
    res = minimize(f_and_df, w0, args=(X, y, lam), method="CG",
                   jac=True, bounds=None)
    return res


# %% Solvers utils


def minibatch_gradient(x, X, y, lam, minibatch):
    # x: evaluate gradient at this value
    M = minibatch.shape[0]
    samples_x = X[minibatch, :]  # matrix
    samples_y = y[minibatch]  # vector
    # in oder to sum over the gradients, regularization term is multiplied by M
    grad_sum = logistic_der(x, samples_x, samples_y, lam * M)  # check
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
    # return np.linalg.norm(grad_k) > tol and nit < max_iter
    # return np.linalg.norm(grad_k) > tol * (1 + fun_k) and nit < max_iter
    return nit < max_iter


# def stopping(nit, max_iter):
    # return nit < max_iter

# %% [1,2,4] SGD-Fixed/Decreasing, SGDM


# SGD-Fixed, SGD-Decreasing, SGDM
def sgd_m(w0, X, y, lam, M, alpha0, beta0, epochs, tol, solv="SGD-Fixed"):
    alpha_seq = np.zeros(epochs)  # one alpha for every epoch
    if solv in ("SGD-Fixed", "SGDM"):
        alpha_seq += alpha0  # same step-size for every epoch
    elif solv == "SGD-Decreasing":
        alpha_seq = alpha0 / (np.arange(alpha_seq.size) + 1)
    N, p = X.shape
    w_seq = np.zeros((epochs + 1, p)) # weights sequence
    w_seq[0, :] = w0
    fun_seq = np.zeros(epochs + 1)  # full objective function sequence
    grad_seq = np.zeros((epochs + 1, p))  # full gradient sequence
    fun_seq[0], grad_seq[0, :] = f_and_df(w0, X, y, lam)
    # funcalls = 1
    # gradcalls = 1
    start = time.time()
    k = 0  # epochs counter
    while stopping(fun_seq[k], grad_seq[k, :], k, epochs, tol):
        minibatches = shuffle_dataset(N, k, M)  # get random minibatches
        y_seq = np.zeros((len(minibatches) + 1, p))  # internal updates
        y_seq[0, :] = w_seq[k]  # y0 = wk
        d_seq = np.zeros((len(minibatches), p))  # internal directions
        for t, minibatch in enumerate(minibatches):  # 0 to N/M-1
            mini_grad = minibatch_gradient(y_seq[t, :], X, y, lam, minibatch)
            # d_seq[t, :] = - mini_grad  # SGD
            # t-1 may work because when t=0 gets the last element that is zero
            d_seq[t, :] = - ((1 - beta0) * mini_grad + beta0 * d_seq[t-1, :])
            y_seq[t+1, :] = y_seq[t, :] + alpha_seq[k] * d_seq[t, :]
        k += 1
        w_seq[k, :] = y_seq[-1, :]  # next weights
        fun_seq[k], grad_seq[k, :] = f_and_df(y_seq[-1, :], X, y, lam)
    end = time.time()
    return OptimizeResult(fun=fun_seq[k], x=w_seq[k, :], jac=grad_seq[k, :],
                          success=(k > 1), solver=solv, fun_per_it=fun_seq,
                          minibatch_size=M, nit=k, runtime=(end - start),
                          step_size=alpha0, momentum=beta0)


# %% [3,5a,5b] SGD-Armijo, MSL-SGDM-C/R


def reset_step(N, alpha, alpha0, M, t):
    # alpha: previous iteration step-size
    opt = 2
    a = 2
    if t == 0:
        return alpha0
    if opt == 0:
        return alpha
    if opt == 1:
        return alpha0
    return alpha * a ** (M / N)


def armijo_condition(x, d, X, y, lam, alpha):
    g = 0.5  # gamma
    fun, jac = f_and_df(x, X, y, lam)
    # grad_norm = np.linalg.norm(jac)
    # thresh = fun - g * alpha * grad_norm ** 2
    thresh = fun + g * alpha * np.dot(jac, d)  # right side
    x_next = x + alpha * d
    fun_next = logistic(x_next, X, y)  # left side
    return fun_next > thresh


def armijo_method(x, d, X, y, lam, alpha, alpha0, M, t):
    # N, p = X.shape
    N = X.shape[0]
    delt = 0.5  # delta factor
    alpha = reset_step(N, alpha, alpha0, M, t) / delt
    # x_next = x + alpha * d
    q = 0  # step-size rejections counter
    while armijo_condition(x, d, X, y, lam, alpha) and q < 20:
        alpha = delt * alpha  # reduce step-size
        # x_next = x + alpha * d
        q += 1
    return alpha#, x_next


def direction_condition(jac, d):
    return np.dot(jac, d) < 0  # True: descent direction


def momentum_correction(beta0, jac, d):
    beta = beta0
    # compute potential next direction
    d_next = - ((1 - beta) * jac + beta * d)
    q = 0  # momentum term rejections counter
    while not direction_condition(jac, d_next) and q < 10:
        # reduce momentum term until descent direction
        beta = 0.5 * beta
        d_next = - ((1 - beta) * jac + beta * d)
        q += 1
    return d_next


def momentum_restart(beta0, jac, d):
    # compute potential next direction
    d_next = - ((1 - beta0) * jac + beta0 * d)
    if direction_condition(jac, d_next):  # descent direction
        return d_next
    return - (1 - beta0) * jac  # restart with d=d0=0


def sgd_sls(w0, X, y, lam, M, alpha0, beta0, epochs, tol, solv="SGD-Armijo"):
    N, p = X.shape  # number of examples and features
    # weights sequence
    w_seq = np.zeros((epochs + 1, p))
    w_seq[0, :] = w0
    # full objective function and full gradient norm sequences
    fun_seq = np.zeros(epochs + 1)
    grad_seq = np.zeros((epochs + 1, p))
    fun_seq[0], grad_seq[0, :] = f_and_df(w0, X, y, lam)
    start = time.time()  # timer
    k = 0  # epochs counter
    while stopping(fun_seq[k], grad_seq[k], k, epochs, tol):
        minibatches = shuffle_dataset(N, k, M)  # get random minibatches
        y_seq = np.zeros((len(minibatches) + 1, p))  # internal updates
        y_seq[0, :] = w_seq[k]  # y0 = wk
        d_seq = np.zeros((len(minibatches), p))  # internal directions
        alpha_seq = np.zeros(len(minibatches))  # step-size per minibatch
        for t, minibatch in enumerate(minibatches):
            mini_grad = minibatch_gradient(y_seq[t, :], X, y, lam, minibatch)
            # when t=0 -> gets a null direction that is the last initialized
            if solv == "SGD-Armijo":
                d_seq[t, :] = - mini_grad
            elif solv == "MSL-SGDM-C":
                # update beta until d_next is descent
                d_seq[t, :] = momentum_correction(beta0, mini_grad, d_seq[t-1, :])
            elif solv == "MSL-SGDM-R":
                # check if d_next is descent
                d_seq[t, :] = momentum_restart(beta0, mini_grad, d_seq[t-1, :])
            # Armijo line search [3,5a,5b]
            alpha_seq[t] = armijo_method(y_seq[t, :], d_seq[t, :], X, y, lam,
                                         alpha_seq[t-1], alpha0, M, t)
            y_seq[t+1, :] = y_seq[t, :] + alpha_seq[t] * d_seq[t, :]
        # Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_seq[-1, :]
        fun_seq[k], grad_seq[k, :] = f_and_df(y_seq[-1, :], X, y, lam)
    end = time.time()  # timer
    return OptimizeResult(fun=fun_seq[k], x=w_seq[k, :], jac=grad_seq[k, :],
                          success=(k > 1), solver=solv, fun_per_it=fun_seq,
                          minibatch_size=M, nit=k, runtime=(end - start),
                          step_size=alpha0, momentum=beta0)
