# -*- coding: utf-8 -*-

# %% Packages
import time
import numpy as np
from scipy.optimize import minimize, OptimizeResult

from solvers_utils import logistic, loss_and_regul, logistic_der, f_and_df_log, logistic_hess

# %% [0] L-BFGS-B / Newton-CG / CG


def l_bfgs_b(w0, X, y, lam):
    res = minimize(f_and_df_log, w0, args=(X, y, lam), method="L-BFGS-B",
                   jac=True, bounds=None)

    return res


def newton_cg(w0, X, y, lam):
    res = minimize(f_and_df_log, w0, args=(X, y, lam), method="Newton-CG",
                   jac=True, hess=logistic_hess, bounds=None)

    return res


def cg(w0, X, y, lam):
    res = minimize(f_and_df_log, w0, args=(X, y, lam), method="CG",
                   jac=True, bounds=None)

    return res


# %% [1,2,4] SGD-Fixed/Decreasing, SGDM


"""
Example:
_minimize_bfgs(fun, x0, args=(), jac=None, callback=None,
            gtol=1e-5, norm=np.inf, eps=_epsilon, maxiter=None,
            disp=False, return_all=False, finite_diff_rel_step=None,
            xrtol=0, c1=1e-4, c2=0.9, 
            hess_inv0=None, **unknown_options):
"""

# SGD-Fixed, SGD-Decreasing, SGDM
def sgd_m(w0, X, y, lam, M, alpha0, beta0, epochs, solver, stop):
    # p = X.shape[1]  # features number

    # allocate sequences
    # w_seq = np.empty((epochs + 1, p))  # weights sequence
    # fun_seq = np.empty(epochs + 1)  # full objective function sequence
    loss_seq = np.empty(epochs + 1)  # loss function sequence
    # grad_seq = np.empty_like(w_seq) # full gradient sequence
    time_seq = np.empty_like(loss_seq)  # time to epoch sequence

    w_k = w0.copy()
    # fun_k, jac_k = f_and_df_log(w_k, X, y, lam)
    loss_k, fun_k = loss_and_regul(w_k, X, y, lam)
    jac_k = logistic_der(w_k, X, y, lam)

    # w_seq[0, :] = w_k.copy()
    loss_seq[0] = loss_k.copy()
    # grad_seq[0, :] = jac_k.copy()
    time_seq[0] = 0

    start = time.time()

    k = 0  # epochs counter
    while stopping(fun_k, jac_k, k, epochs, criterion=stop):
        # split dataset indices randomly
        minibatches = shuffle_dataset(X.shape[0], k, M)

        z_t = w_k.copy()  # starting model
        d_t = np.zeros_like(z_t)  # initialize direction

        # fixed or decreasing step-size
        alpha = select_step1(solver, alpha0, k)

        for t, minibatch in enumerate(minibatches):  # 0 to N/M-1
            # compute gradient on the considered minibatch
            jac_t = batch_jac(z_t, X, y, lam, minibatch)

            # direction: anti-gradient or momentum
            d_t = -((1 - beta0) * jac_t + beta0 * d_t)

            # update model
            z_t += alpha * d_t

        k += 1

        w_k = z_t.copy()
        # fun_k, jac_k = f_and_df_log(w_k, X, y, lam)
        loss_k, fun_k = loss_and_regul(w_k, X, y, lam)
        jac_k = logistic_der(w_k, X, y, lam)

        # w_seq[k, :] = w_k.copy()
        loss_seq[k] = loss_k.copy()
        # grad_seq[k, :] = jac_k.copy()
        time_seq[k] = time.time() - start  # time to epoch

    result = OptimizeResult(fun=fun_k.copy(), x=w_k.copy(), jac=jac_k.copy(),
                            success=(k > 1), solver=solver, minibatch_size=M,
                            nit=k, runtime=time_seq[k], time_per_epoch=time_seq,
                            step_size=alpha0, momentum=beta0,
                            loss_per_epoch=loss_seq)
    return result


# %% [3,5a,5b] SGD-Armijo, MSL-SGDM-C/R


def sgd_sls(w0, X, y, lam, M, alpha0, beta0, epochs, solver, stop):
    # p = X.shape[1]  # features number

    # allocate sequences
    # w_seq = np.empty((epochs + 1, p))  # weights sequence
    # fun_seq = np.empty(epochs + 1)  # full objective function sequence
    loss_seq = np.empty(epochs + 1)  # loss function sequence
    # grad_seq = np.empty_like(w_seq) # full gradient sequence
    time_seq = np.empty_like(loss_seq)  # time to epoch sequence

    w_k = w0.copy()
    # fun_k, jac_k = f_and_df_log(w_k, X, y, lam)
    loss_k, fun_k = loss_and_regul(w_k, X, y, lam)
    jac_k = logistic_der(w_k, X, y, lam)

    # w_seq[0, :] = w_k.copy()
    loss_seq[0] = loss_k.copy()
    # grad_seq[0, :] = jac_k.copy()
    time_seq[0] = 0

    start = time.time()

    k = 0
    while stopping(fun_k, jac_k, k, epochs, criterion=stop):
        # split dataset indices randomly
        minibatches = shuffle_dataset(X.shape[0], k, M)  # get random minibatches

        z_t = w_k.copy()  # starting model
        d_t = np.zeros_like(z_t)  # initialize direction

        # initialize iterations step-size
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
        # fun_k, jac_k = f_and_df_log(w_k, X, y, lam)
        loss_k, fun_k = loss_and_regul(w_k, X, y, lam)
        jac_k = logistic_der(w_k, X, y, lam)

        # w_seq[k, :] = w_k.copy()
        loss_seq[k] = loss_k.copy()
        # grad_seq[k, :] = jac_k.copy()
        time_seq[k] = time.time() - start  # time to epoch

    result = OptimizeResult(fun=fun_k.copy(), x=w_k.copy(), jac=jac_k.copy(),
                            success=(k > 1), solver=solver, minibatch_size=M,
                            nit=k, runtime=time_seq[k], time_per_epoch=time_seq,
                            step_size=alpha0, momentum=beta0,
                            loss_per_epoch=loss_seq)
    return result


# %% utils


def stopping(fun_k, grad_k, nit, max_iter, criterion):
    # fun and grad already evaluated
    tol = 1e-3
    stop = False

    if criterion == 0:
        stop = nit < max_iter

    if criterion == 1:
        stop = (np.linalg.norm(grad_k) > tol) and (nit < max_iter)

    if criterion == 2:
        # return (np.linalg.norm(grad_k, np.inf) > tol) and (nit < max_iter)
        stop = (np.linalg.norm(grad_k) > tol * (1 + fun_k)) and (nit < max_iter)

    return stop


def batch_jac(z, X, y, lam, minibatch):
    # z: gradient w.r.t.
    # minibatch: array

    samples_x = X[minibatch, :].copy()  # matrix
    samples_y = y[minibatch].copy()     # vector

    # compute minibatch gradient
    grad_sum = logistic_der(z, samples_x, samples_y, lam)

    return grad_sum


def shuffle_dataset(N, k, M):
    batch = np.arange(N)  # dataset indices, reset every epoch

    rng = np.random.default_rng(k)  # set different seed every epoch
    rng.shuffle(batch)              # shuffle indices

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


# def select_direction1(beta0, jac, d, t):
#     # d: previous direction

#     if t == 0:
#         return -(1 - beta0) * jac

#     return -((1 - beta0) * jac + beta0 * d)


# %% utils sls


def select_direction2(solver, beta0, jac, d):
    # d: previous direction

    d_next = np.empty_like(d)

    if solver == "SGD-Armijo":
        # set negative gradient as the direction
        d_next = - jac

    elif solver == "MSL-SGDM-C":
        # update momentum until the direction is descent
        d_next = momentum_correction(beta0, jac, d)

    elif solver == "MSL-SGDM-R":
        # if not descent set direction to damped negative gradient
        d_next = momentum_restart(beta0, jac, d)

    return d_next


def momentum_correction(beta0, jac, d):
    beta = beta0  # initial momentum
    delta = 0.75  # momentum damping factor

    d_next = - ((1 - beta) * jac + beta * d)  # starting direction

    q = 0  # momentum term rejections counter
    while (not np.dot(jac, d_next) < 0) and (q < 20):
        beta = delta * beta  # reduce momentum term

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
    opt = 2  # step-size restart rule
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
    delta = 0.75  # step-size damping factor
    gamma = 0.1  # Armijo condition coefficient

    # reset step-size
    alpha = reset_step(X.shape[0], alpha_old, alpha_init, M, t) / delta

    fun, jac = f_and_df_log(z, X, y, lam)  # w.r.t. z
    z_next = z + alpha * d  # update model with starting step-size
    fun_next = logistic(z_next, X, y, lam)  # w.r.t. potential next z

    # general Armijo condition
    condition = fun_next - (fun + gamma * alpha * np.dot(jac, d))

    q = 0  # step-size rejections counter
    while (not condition <= 0) and (q < 20):
        alpha = delta * alpha  # reduce step-size

        z_next = z + alpha * d  # update model with reduced step-size
        fun_next = logistic(z_next, X, y, lam)  # w.r.t. potential next z

        # general Armijo condition once more
        condition = fun_next - (fun + 0.1 * alpha * np.dot(jac, d))

        q += 1

    return alpha, z_next
