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

# TODO: time per epoch

# SGD-Fixed, SGD-Decreasing, SGDM
def sgd_m(w0, X, y, lam, M, alpha0, beta0, epochs, solver):
    # p = X.shape[1]  # features number

    # w_seq = np.empty((epochs + 1, p))  # weights sequence
    fun_seq = np.empty(epochs + 1)  # full objective function sequence
    # grad_seq = np.empty_like(w_seq) # full gradient sequence

    w_k = w0.copy()
    # w_seq[0, :] = w_k.copy()

    fun_k, jac_k = f_and_df(w_k, X, y, lam)
    fun_seq[0] = fun_k.copy()
    # grad_seq[0, :] = jac_k.copy()

    start = time.time()
    k = 0  # epochs counter
    while stopping(fun_k, jac_k, k, epochs):
        # split dataset indices randomly
        minibatches = shuffle_dataset(X.shape[0], k, M)

        z_t = w_k.copy()  # starting model
        d_t = np.empty_like(z_t)  # starting direction

        # constant or decreasing step-size
        alpha = select_step1(solver, alpha0, k)

        for t, minibatch in enumerate(minibatches):  # 0 to N/M-1
            # compute gradient on the considered minibatch
            jac_t = batch_jac(z_t, X, y, lam, minibatch)

            # direction: anti-gradient or momentum
            d_t = select_direction1(beta0, jac_t, d_t, t)

            # update model
            z_t += alpha * d_t

        k += 1

        w_k = z_t.copy()
        # w_seq[k, :] = w_k.copy()

        fun_k, jac_k = f_and_df(z_t, X, y, lam)
        fun_seq[k] = fun_k.copy()
        # grad_seq[k, :] = jac_k.copy()

    end = time.time()

    result = OptimizeResult(fun=fun_k.copy(), x=w_k.copy(),
                            jac=jac_k.copy(), success=(k > 1),
                            solver=solver, minibatch_size=M,
                            nit=k, runtime=(end - start), step_size=alpha0,
                            momentum=beta0, fun_per_it=fun_seq)
    return result


# %% [3,5a,5b] SGD-Armijo, MSL-SGDM-C/R


def sgd_sls(w0, X, y, lam, M, alpha0, beta0, epochs, solver):
    # p = X.shape[1]  # features number

    # w_seq = np.empty((epochs + 1, p))  # weights sequence
    fun_seq = np.empty(epochs + 1)  # full objective function sequence
    # grad_seq = np.empty_like(w_seq) # full gradient sequence

    w_k = w0.copy()
    # w_seq[0, :] = w_k.copy()

    fun_k, jac_k = f_and_df(w_k, X, y, lam)
    fun_seq[0] = fun_k.copy()
    # grad_seq[0, :] = jac_k.copy()

    start = time.time()
    k = 0
    while stopping(fun_k, jac_k, k, epochs):
        # split dataset indices randomly
        minibatches = shuffle_dataset(X.shape[0], k, M)  # get random minibatches

        z_t = w_k.copy()  # starting model
        d_t = np.empty_like(z_t)  # starting direction

        # initialize iterations' step-size
        alpha_t = alpha0

        for t, minibatch in enumerate(minibatches):
            # compute gradient on the considered minibatch
            jac_t = batch_jac(z_t, X, y, lam, minibatch)

            # direction: anti-gradient, momentum correction or restart
            d_t = select_direction2(solver, beta0, jac_t, d_t)

            # Armijo (stochastic) line search and model update
            alpha_t, z_t = armijo_method(
                z_t, d_t, X, y, lam, alpha_t, alpha0, M, t)

        k += 1

        w_k = z_t.copy()
        # w_seq[k, :] = w_k.copy()

        fun_k, jac_k = f_and_df(z_t, X, y, lam)
        fun_seq[k] = fun_k.copy()
        # grad_seq[k, :] = jac_k.copy()

    end = time.time()

    result = OptimizeResult(fun=fun_k.copy(), x=w_k.copy(),
                            jac=jac_k.copy(), success=(k > 1),
                            solver=solver, minibatch_size=M,
                            nit=k, runtime=(end - start), step_size=alpha0,
                            momentum=beta0, fun_per_it=fun_seq)
    return result


# %% utils


def stopping(fun_k, grad_k, nit, max_iter):
    # fun and grad already evaluated
    # tol = 1e-2
    # return (np.linalg.norm(grad_k) > tol) and (nit < max_iter)
    # return (np.linalg.norm(grad_k) > tol * (1 + fun_k)) and (nit < max_iter)
    # return (np.linalg.norm(grad_k, np.inf) > tol) and (nit < max_iter)
    return (nit < max_iter)


def batch_jac(z, X, y, lam, minibatch):
    # z: gradient w.r.t.
    # minibatch: array

    samples_x = X[minibatch, :].copy()  # matrix
    samples_y = y[minibatch].copy()  # vector

    # compute minibatch gradient
    grad_sum = logistic_der(z, samples_x, samples_y, lam)

    return grad_sum


def shuffle_dataset(N, k, M):
    batch = np.arange(N)  # dataset indices, reset every epoch

    rng = np.random.default_rng(k)  # set different seed every epoch
    rng.shuffle(batch)  # shuffle indices

    # array_split is expensive, consider another strategy
    minibatches = np.array_split(batch, N / M)  # create the minibatches

    return minibatches  # list of numpy.ndarray


# %% utils basic


def select_step1(solver, alpha, k):
    if solver in ("SGD-Fixed", "SGDM"):
        pass

    elif solver == "SGD-Decreasing":
        alpha = alpha / (k + 1)

    return alpha


def select_direction1(beta0, jac, d, t):
    if t == 0:
        return -(1 - beta0) * jac

    return -((1 - beta0) * jac + beta0 * d)


# %% utils sls


def select_direction2(solver, beta0, jac, d):
    if solver == "SGD-Armijo":
        d = - jac  # anti-gradient

    elif solver == "MSL-SGDM-C":
        d = momentum_correction(beta0, jac, d)

    elif solver == "MSL-SGDM-R":
        d = momentum_restart(beta0, jac, d)

    return d


def momentum_correction(beta0, jac, d):
    beta = beta0

    d_next = - ((1 - beta) * jac + beta * d)  # starting direction

    q = 0  # momentum term rejections counter
    while not np.dot(jac, d_next) < 0 and q < 10:
        beta = 0.5 * beta  # reduce momentum term

        # update direction with reduced momentum term
        d_next = - ((1 - beta) * jac + beta * d)

        q += 1

    return d_next


def momentum_restart(beta0, jac, d):
    d_next1 = - (1 - beta0) * jac
    d_next2 = - beta0 * d

    if np.dot(jac, d_next1 + d_next2) < 0:  # if descent direction
        d = d_next1 + d_next2

    else:  # restart with d=d0=0
        d = d_next1

    return d


def reset_step(N, alpha, alpha0, M, t):
    # alpha: previous iteration step-size
    # alpha0: initial step-size
    opt = 2
    a = 2

    if t == 0 or opt == 1:
        alpha = alpha0

    elif opt == 0:
        pass

    elif opt == 2:
        alpha = alpha * a**(M / N)

    return alpha


def armijo_method(z, d, X, y, lam, alpha_old, alpha_init, M, t):
    # returns: selected step-size and model update
    delta = 0.9  # step-size damping factor

    # reset step-size
    alpha = reset_step(X.shape[0], alpha_old, alpha_init, M, t) / delta

    fun, jac = f_and_df(z, X, y, lam)  # w.r.t. z
    z_next = z + alpha * d  # update model with starting step-size
    fun_next = logistic(z_next, X, y, lam)  # w.r.t. potential next z

    # general Armijo condition
    condition = fun_next - (fun + 0.5 * alpha * np.dot(jac, d))

    q = 0  # step-size rejections counter
    while not condition <= 0 and q < 50:
        alpha = delta * alpha  # reduce step-size

        z_next = z + alpha * d  # update model with reduced step-size
        fun_next = logistic(z_next, X, y, lam)  # w.r.t. potential next z

        # general Armijo condition once more
        condition = fun_next - (fun + 0.1 * alpha * np.dot(jac, d))

        q += 1

    return alpha, z_next
