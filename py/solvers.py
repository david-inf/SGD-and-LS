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


def minibatch_gradient(z, X, y, lam, minibatch):
    # z: evaluate gradient at this value
    # minibatch: array
    M = minibatch.shape[0]
    samples_x = X[minibatch, :].copy()  # matrix
    samples_y = y[minibatch].copy()  # vector
    # in oder to sum over the gradients, regularization term is multiplied by M
    grad_sum = logistic_der(z, samples_x, samples_y, lam * M)  # check
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
    # return (np.linalg.norm(grad_k) > tol) and (nit < max_iter)
    # return (np.linalg.norm(grad_k) > tol * (1 + fun_k)) and (nit < max_iter)
    return (nit < max_iter)


# def stopping(nit, max_iter):
    # return nit < max_iter

# %% [1,2,4] SGD-Fixed/Decreasing, SGDM


# SGD-Fixed, SGD-Decreasing, SGDM
def sgd_m(w0, X, y, lam, M, alpha0, beta0, epochs, tol, solv):
    N, p = X.shape
    w_seq = np.zeros((epochs + 1, p))  # weights sequence
    w_seq[0, :] = w0
    fun_seq = np.zeros(epochs + 1)  # full objective function sequence
    grad_seq = np.zeros((epochs + 1, p))  # full gradient sequence
    fun_seq[0], grad_seq[0, :] = f_and_df(w0, X, y, lam)
    start = time.time()
    k = 0
    while stopping(fun_seq[k], grad_seq[k, :], k, epochs, tol):
        minibatches = shuffle_dataset(N, k, M)  # get random minibatches
        y_seq = np.zeros((len(minibatches) + 1, p))  # internal updates
        y_seq[0, :] = w_seq[k, :].copy()  # y0 = wk
        # z = w_seq[k, :].copy()
        d_seq = np.zeros((len(minibatches), p))  # internal directions
        alpha = None
        if solv in ("SGD-Fixed", "SGDM"):
            alpha = alpha0
        elif solv == "SGD-Decreasing":
            alpha = alpha0 / (k + 1)
        for t, minibatch in enumerate(minibatches):  # 0 to N/M-1
            mini_grad = minibatch_gradient(y_seq[t, :], X, y, lam, minibatch)
            # mini_grad = minibatch_gradient(z, X, y, lam, minibatch)
            # t-1 may work because when t=0 gets the last element that is zero
            d_seq[t, :] = - ((1 - beta0) * mini_grad + beta0 * d_seq[t-1, :])
            y_seq[t+1, :] = y_seq[t, :] + alpha * d_seq[t, :]
            # z += alpha * d_seq[t, :]
        k += 1
        w_seq[k, :] = y_seq[-1, :].copy()  # next weights
        # w_seq[k, :] = z
        fun_seq[k], grad_seq[k, :] = f_and_df(y_seq[-1, :], X, y, lam)
        # fun_seq[k], grad_seq[k, :] = f_and_df(z, X, y, lam)
    end = time.time()
    result = OptimizeResult(fun=fun_seq[k].copy(), x=w_seq[k, :].copy(), 
                            jac=grad_seq[k, :].copy(), success=(k > 1),
                            solver=solv, fun_per_it=fun_seq, minibatch_size=M,
                            nit=k, runtime=(end - start), step_size=alpha0,
                            momentum=beta0)
    return result


# %% [3,5a,5b] SGD-Armijo, MSL-SGDM-C/R


def sgd_sls(w0, X, y, lam, M, alpha0, beta0, epochs, tol, solv="SGD-Armijo"):
    N, p = X.shape  # number of examples and features
    w_seq = np.zeros((epochs + 1, p))  # weights sequence
    w_seq[0, :] = w0
    fun_seq = np.zeros(epochs + 1)  # full objective function sequence
    grad_seq = np.zeros((epochs + 1, p))  # full gradient sequence
    fun_seq[0], grad_seq[0, :] = f_and_df(w0, X, y, lam)
    start = time.time()
    k = 0
    while stopping(fun_seq[k], grad_seq[k, :], k, epochs, tol):
        minibatches = shuffle_dataset(N, k, M)  # get random minibatches
        y_seq = np.zeros((len(minibatches) + 1, p))  # internal updates
        y_seq[0, :] = w_seq[k, :].copy()  # y0 = wk
        d_seq = np.zeros((len(minibatches), p))  # internal directions
        alpha_seq = np.zeros(len(minibatches))  # step-size per minibatch
        for t, minibatch in enumerate(minibatches):
            mini_grad = minibatch_gradient(y_seq[t, :], X, y, lam, minibatch)
            # when t=0 -> gets a null direction that is the last initialized
            if solv == "SGD-Armijo":
                d_seq[t, :] = - mini_grad
            elif solv == "MSL-SGDM-C":
                d_seq[t, :] = momentum_correction(
                    beta0, mini_grad, d_seq[t-1, :])
            elif solv == "MSL-SGDM-R":
                d_seq[t, :] = momentum_restart(
                    beta0, mini_grad, d_seq[t-1, :])
            # Armijo line search
            alpha_start = reset_step(N, alpha_seq[t-1], alpha0, M, t) / 0.5
            alpha_seq[t], y_seq[t+1, :] = armijo_method(
                y_seq[t, :], d_seq[t, :], X, y, lam, alpha_start)
        # Update sequence, objective function and gradient norm
        k += 1
        w_seq[k, :] = y_seq[-1, :].copy()
        fun_seq[k], grad_seq[k, :] = f_and_df(y_seq[-1, :], X, y, lam)
    end = time.time()
    result = OptimizeResult(fun=fun_seq[k].copy(), x=w_seq[k, :].copy(), 
                            jac=grad_seq[k, :].copy(), success=(k > 1),
                            solver=solv, fun_per_it=fun_seq, minibatch_size=M,
                            nit=k, runtime=(end - start), step_size=alpha0,
                            momentum=beta0)
    return result


# %% SLS utils


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


def armijo_condition(x, d, x_next, X, y, lam, alpha):
    fun, jac = f_and_df(x, X, y, lam)
    thresh = fun + 0.25 * alpha * np.dot(jac, d)  # right side
    fun_next = logistic(x_next, X, y)  # left side
    return fun_next <= thresh


def armijo_method(x, d, X, y, lam, alpha0):
    alpha = alpha0
    x_next = x + alpha * d
    q = 0  # step-size rejections counter
    while not armijo_condition(x, d, x_next, X, y, lam, alpha) and q < 10:
        alpha = 0.5 * alpha  # reduce step-size
        x_next = x + alpha * d
        q += 1
    return alpha, x_next


def momentum_correction(beta0, jac, d):
    beta = beta0
    d_next = - ((1 - beta) * jac + beta * d)
    q = 0  # momentum term rejections counter
    while not np.dot(jac, d_next) < 0 and q < 10:
        beta = 0.5 * beta  # reduce momentum term
        d_next = - ((1 - beta) * jac + beta * d)
        q += 1
    return d_next


def momentum_restart(beta0, jac, d):
    d_next1 = - (1 - beta0) * jac
    d_next2 = - beta0 * d
    if np.dot(jac, d_next1 + d_next2) < 0:  # if descent direction
        return d_next1 + d_next2
    return d_next1  # restart with d=d0=0
