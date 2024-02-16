# -*- coding: utf-8 -*-

# %% Packages
import time
import numpy as np
from scipy.optimize import minimize, OptimizeResult
from solvers_utils import logistic, logistic_der, f_and_df, logistic_hess

# %% [0] L-BFGS-B / Newton-CG / CG


def l_bfgs_b(w0, X, y, lam):
    res = minimize(f_and_df, w0, args=(X, y, lam), method="L-BFGS-B",
                   jac=True, bounds=None)
    return res


def newton_cg(w0, X, y, lam):
    res = minimize(f_and_df, w0, args=(X, y, lam), method="Newton-CG",
                   jac=True, hess=logistic_hess, bounds=None)
    return res


def cg(w0, X, y, lam):
    res = minimize(f_and_df, w0, args=(X, y, lam), method="CG",
                   jac=True, bounds=None)
    return res


# %% [1,2,4] SGD-Fixed/Decreasing, SGDM


# SGD-Fixed, SGD-Decreasing, SGDM
def sgd_m(w0, X, y, lam, M, alpha0, beta0, epochs, solver):
    p = X.shape[1]  # features number

    w_seq = np.empty((epochs + 1, p))  # weights sequence
    fun_seq = np.empty(epochs + 1)  # full objective function sequence
    grad_seq = np.empty_like(w_seq) # full gradient sequence

    w_seq[0, :] = w0
    fun_seq[0], grad_seq[0, :] = f_and_df(w0, X, y, lam)

    start = time.time()
    k = 0
    while stopping(fun_seq[k], grad_seq[k, :], k, epochs):
        minibatches = shuffle_dataset(X.shape[0], k, M)  # get random minibatches
        z_t = w_seq[k, :].copy()  # start model
        d_t = np.empty_like(w0)  # start direction

        alpha = select_step1(solver, alpha0, k)  # learning rate

        for t, minibatch in enumerate(minibatches):  # 0 to N/M-1
            jac_t = batch_jac(z_t, X, y, lam, minibatch)  # minibatch gradient
            d_t = basic_direction(beta0, jac_t, d_t, t)  # update direction
            z_t += alpha * d_t  # update model

        k += 1
        w_seq[k, :] = z_t
        fun_seq[k], grad_seq[k, :] = f_and_df(z_t, X, y, lam)

    end = time.time()

    result = OptimizeResult(fun=fun_seq[k].copy(), x=w_seq[k, :].copy(), 
                            jac=grad_seq[k, :].copy(), success=(k > 1),
                            solver=solver, fun_per_it=fun_seq, minibatch_size=M,
                            nit=k, runtime=(end - start), step_size=alpha0,
                            momentum=beta0)
    return result


# %% [3,5a,5b] SGD-Armijo, MSL-SGDM-C/R


def sgd_sls(w0, X, y, lam, M, alpha0, beta0, epochs, solver):
    N, p = X.shape  # features and samples number

    w_seq = np.empty((epochs + 1, p))  # weights sequence
    fun_seq = np.empty(epochs + 1)  # full objective function sequence
    grad_seq = np.empty_like(w_seq) # full gradient sequence

    w_seq[0, :] = w0
    fun_seq[0], grad_seq[0, :] = f_and_df(w0, X, y, lam)

    start = time.time()
    k = 0
    while stopping(fun_seq[k], grad_seq[k, :], k, epochs):
        minibatches = shuffle_dataset(N, k, M)  # get random minibatches

        # y_seq = np.zeros((len(minibatches) + 1, p))  # internal updates
        # y_seq[0, :] = w_seq[k, :].copy()  # y0 = wk
        # d_seq = np.zeros((len(minibatches), p))  # internal directions
        z_t = w_seq[k, :].copy()  # start model
        d_t = np.empty_like(w0)  # start direction

        # alpha_seq = np.zeros(len(minibatches))  # step-size per minibatch
        alpha_t = alpha0

        for t, minibatch in enumerate(minibatches):
            # jac_t = batch_jac(y_seq[t, :], X, y, lam, minibatch)
            jac_t = batch_jac(z_t, X, y, lam, minibatch)
            d_t = select_step2(solver, beta0, jac_t, d_t)

            # Armijo line search
            # alpha_start = reset_step(N, alpha_seq[t-1], alpha0, M, t) / 0.5
            alpha_start = reset_step(N, alpha_t, alpha0, M, t) / 0.5
            # alpha_seq[t], y_seq[t+1, :] = armijo_method(
            #     y_seq[t, :], d_seq[t, :], X, y, lam, alpha_start)
            alpha_t, z_t = armijo_method(
                z_t, d_t, X, y, lam, alpha_start)

        k += 1
        # w_seq[k, :] = y_seq[-1, :].copy()
        w_seq[k, :] = z_t
        # fun_seq[k], grad_seq[k, :] = f_and_df(y_seq[-1, :], X, y, lam)
        fun_seq[k], grad_seq[k, :] = f_and_df(z_t, X, y, lam)

    end = time.time()

    result = OptimizeResult(fun=fun_seq[k].copy(), x=w_seq[k, :].copy(), 
                            jac=grad_seq[k, :].copy(), success=(k > 1),
                            solver=solver, fun_per_it=fun_seq, minibatch_size=M,
                            nit=k, runtime=(end - start), step_size=alpha0,
                            momentum=beta0)
    return result


# %% utils


def stopping(fun_k, grad_k, nit, max_iter):
    # fun and grad already evaluated
    # tol=1e-2
    # return (np.linalg.norm(grad_k) > tol) and (nit < max_iter)
    # return (np.linalg.norm(grad_k) > tol * (1 + fun_k)) and (nit < max_iter)
    return (nit < max_iter)


def select_step1(solver, alpha0, k):
    if solver in ("SGD-Fixed", "SGDM"):
        return alpha0
    if solver == "SGD-Decreasing":
        return alpha0 / (k + 1)


def select_step2(solver, beta0, jac, d):
    # when t=0 -> gets a null direction that is the last initialized
    if solver == "SGD-Armijo":
        # d_seq[t, :] = - jac_t
        d = - jac
    elif solver == "MSL-SGDM-C":
        # d_seq[t, :] = momentum_correction(
            # beta0, jac_t, d_seq[t-1, :])
        d = momentum_correction(beta0, jac, d)
    elif solver == "MSL-SGDM-R":
        # d_seq[t, :] = momentum_restart(
            # beta0, jac_t, d_seq[t-1, :])
        d = momentum_restart(beta0, jac, d)
    return d


def basic_direction(beta0, jac, d, t):
    if t == 0:
        return -(1 - beta0) * jac
    return -((1 - beta0) * jac + beta0 * d)


def batch_jac(z, X, y, lam, minibatch):
    # z: gradient w.r.t.
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
    fun_next = logistic(x_next, X, y, lam)  # left side
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
