# -*- coding: utf-8 -*-

# %% Packages
import time
import numpy as np
from scipy.optimize import OptimizeResult

# %% [1,2,4] SGD-Fixed/Decreasing, SGDM


def sgd_m(w0, X, y, lam, M, alpha0, beta0, epochs, solver, stop,
          fun, jac, f_and_df):
    """
    SGD-Fixed, SGD-Decreasing, SGDM

    fun, jac, f_and_df: callables
    """

    # p = w0.size
    # w_seq = np.empty((epochs + 1, p))  # weights sequence
    fun_seq = np.empty(epochs + 1)       # full objective function sequence
    # grad_seq = np.empty_like(w_seq)    # full gradient sequence
    time_seq = np.empty_like(fun_seq)    # time per epoch sequence

    w_k = w0.copy()                           # starting solution
    # fun_k = fun(w_k, X, y, lam)             # w.r.t. starting solution
    # grad_k = jac(w_k, X, y, lam             # w.r.t. starting solution
    fun_k, grad_k = f_and_df(w_k, X, y, lam)  # w.r.t. starting solution

    # w_seq[0, :] = w_k.copy()        # add starting solution
    fun_seq[0] = fun_k.copy()         # add evaluation
    # grad_seq[0, :] = grad_k.copy()  # add evaluation
    time_seq[0] = 0                   # count from 0

    start = time.time()  # start time counter

    k = 0  # epochs counter
    while stopping(fun_k, grad_k, k, epochs, criterion=stop):
        # split dataset indices randomly
        minibatches = shuffle_dataset(y.size, k, M)

        z_t = w_k.copy()          # starting model
        d_t = np.zeros_like(z_t)  # initialize direction

        # fixed or decreasing step-size
        alpha = select_step(solver, alpha0, k)

        for _, minibatch in enumerate(minibatches):  # 0 to N/M-1
            # compute gradient on the considered minibatch
            grad_t = batch_grad(jac, z_t, X, y, lam, minibatch)

            # direction: anti-gradient or damped
            d_t = -((1 - beta0) * grad_t + beta0 * d_t)

            # update model
            z_t += alpha * d_t

        k += 1

        w_k = z_t.copy()                          # solution found
        # fun_k = fun(w_k, X, y, lam)             # w.r.t. last solution found
        # grad_k = jac(w_k, X, y, lam)            # w.r.t. last solution found
        fun_k, grad_k = f_and_df(w_k, X, y, lam)  # w.r.t. last solution found

        # w_seq[k, :] = w_k.copy()         # add last solution found
        fun_seq[k] = fun_k.copy()          # add evaluation
        # grad_seq[k, :] = grad_k.copy()   # add evaluation
        time_seq[k] = time.time() - start  # time per epoch

    result = OptimizeResult(fun=fun_k.copy(), x=w_k.copy(), jac=grad_k.copy(),
                            success=(k > 1), solver=solver, minibatch_size=M,
                            nit=k, runtime=time_seq[k], time_per_epoch=time_seq,
                            step_size=alpha0, momentum=beta0,
                            fun_per_epoch=fun_seq)

    return result


# %% [3,5a,5b] SGD-Armijo, MSL-SGDM-C/R


def sgd_sls(w0, X, y, lam, M, alpha0, beta0, epochs, solver, stop,
            fun, jac, f_and_df):
    """
    SGD-Armijo, MSL-SGDM-C, MSL-SGDM-R

    fun, jac, f_and_df: callables
    """

    # p = w0.size
    # w_seq = np.empty((epochs + 1, p))  # weights sequence
    fun_seq = np.empty(epochs + 1)       # full objective function sequence
    # grad_seq = np.empty_like(w_seq)    # full gradient sequence
    time_seq = np.empty_like(fun_seq)    # time per epoch sequence

    w_k = w0.copy()                           # starting solution
    # fun_k = fun(w_k, X, y, lam)             # w.r.t. starting solution
    # grad_k = jac(w_k, X, y, lam             # w.r.t. starting solution
    fun_k, grad_k = f_and_df(w_k, X, y, lam)  # w.r.t. starting solution

    # w_seq[0, :] = w_k.copy()        # add starting solution
    fun_seq[0] = fun_k.copy()         # add evaluation
    # grad_seq[0, :] = grad_k.copy()  # add evaluation
    time_seq[0] = 0                   # count from 0

    start = time.time()  # start time counter

    k = 0
    while stopping(fun_k, grad_k, k, epochs, criterion=stop):
        # split dataset indices randomly
        minibatches = shuffle_dataset(y.size, k, M)

        z_t = w_k.copy()          # starting model
        d_t = np.zeros_like(z_t)  # initialize direction

        # initialize iterations step-size
        alpha_t = alpha0

        for t, minibatch in enumerate(minibatches):
            # compute gradient on the considered minibatch
            grad_t = batch_grad(jac, z_t, X, y, lam, minibatch)

            # direction: anti-gradient, momentum correction or restart
            d_t = select_direction(solver, beta0, grad_t, d_t)

            # Armijo (stochastic) line search and model update
            alpha_t, z_t = armijo_method(
                z_t, d_t, X, y, lam, alpha_t, alpha0, M, t, fun, f_and_df)

        k += 1

        w_k = z_t.copy()                          # solution found
        # fun_k = fun(w_k, X, y, lam)             # w.r.t. last solution found
        # grad_k = jac(w_k, X, y, lam)            # w.r.t. last solution found
        fun_k, grad_k = f_and_df(w_k, X, y, lam)  # w.r.t. last solution found

        # w_seq[k, :] = w_k.copy()         # add last solution found
        fun_seq[k] = fun_k.copy()          # add evaluation
        # grad_seq[k, :] = grad_k.copy()   # add evaluation
        time_seq[k] = time.time() - start  # time per epoch

    result = OptimizeResult(fun=fun_k.copy(), x=w_k.copy(), jac=grad_k.copy(),
                            success=(k > 1), solver=solver, minibatch_size=M,
                            nit=k, runtime=time_seq[k], time_per_epoch=time_seq,
                            step_size=alpha0, momentum=beta0,
                            fun_per_epoch=fun_seq)

    return result


# %% One function for all solvers


def sgd(w0, X, y, lam, M, alpha0, beta0, epochs, solver, stop,
        fun, jac, f_and_df):
    """
    solver:
        SGD-Fixed, SGD-Decreasing, SGDM, SGD-Armijo, MSL-SGDM-C, MSL-SGDM-R
    fun, jac, f_and_df: callables
    """

    # p = w0.size
    # w_seq = np.empty((epochs + 1, p))  # weights sequence
    fun_seq = np.empty(epochs + 1)       # full objective function sequence
    # grad_seq = np.empty_like(w_seq)    # full gradient sequence
    time_seq = np.empty_like(fun_seq)    # time per epoch sequence

    w_k = w0.copy()                           # starting solution
    # fun_k = fun(w_k, X, y, lam)             # w.r.t. starting solution
    # grad_k = jac(w_k, X, y, lam             # w.r.t. starting solution
    fun_k, grad_k = f_and_df(w_k, X, y, lam)  # w.r.t. starting solution

    # w_seq[0, :] = w_k.copy()        # add starting solution
    fun_seq[0] = fun_k.copy()         # add evaluation
    # grad_seq[0, :] = grad_k.copy()  # add evaluation
    time_seq[0] = 0                   # count from 0

    start = time.time()  # start time counter

    k = 0  # epochs counter
    while stopping(fun_k, grad_k, k, epochs, criterion=stop):
        # split dataset indices randomly
        minibatches = shuffle_dataset(y.size, k, M)

        z_t = w_k.copy()          # starting model
        d_t = np.zeros_like(z_t)  # initialize direction

        # fixed or decreasing step-size
        alpha_t = select_step(solver, alpha0, k)

        for t, minibatch in enumerate(minibatches):  # 0 to N/M-1
            # compute gradient on the considered minibatch
            grad_t = batch_grad(jac, z_t, X, y, lam, minibatch)

            # direction: anti-gradient or damped
            d_t = select_direction(solver, beta0, grad_t, d_t)

            if solver in ("SGD-Fixed", "SGD-Decreasing", "SGDM"):
                # weights update without line search update
                z_t += alpha_t * d_t

            elif solver in ("SGD-Armijo", "MSL-SGDM-C", "MSL-SGDM-R"):
                # weights update with line search
                alpha_t, z_t = armijo_method(
                    z_t, d_t, X, y, lam, alpha_t, alpha0, M, t, fun, f_and_df)

        k += 1

        w_k = z_t.copy()                          # solution found
        # fun_k = fun(w_k, X, y, lam)             # w.r.t. last solution found
        # grad_k = jac(w_k, X, y, lam)            # w.r.t. last solution found
        fun_k, grad_k = f_and_df(w_k, X, y, lam)  # w.r.t. last solution found

        # w_seq[k, :] = w_k.copy()         # add last solution found
        fun_seq[k] = fun_k.copy()          # add evaluation
        # grad_seq[k, :] = grad_k.copy()   # add evaluation
        time_seq[k] = time.time() - start  # time per epoch

    result = OptimizeResult(fun=fun_k.copy(), x=w_k.copy(), jac=grad_k.copy(),
                            success=(k > 1), solver=solver, minibatch_size=M,
                            nit=k, runtime=time_seq[k], time_per_epoch=time_seq,
                            step_size=alpha0, momentum=beta0,
                            fun_per_epoch=fun_seq)

    return result


# %% utils


def shuffle_dataset(N, k, M):
    """
    N: dataset size
    k: epochs counter
    M: minibatch size
    """
    batch = np.arange(N)  # dataset indices, reset every epoch

    rng = np.random.default_rng(k)  # set different seed every epoch
    rng.shuffle(batch)              # shuffle indices

    # array_split is expensive, consider another strategy
    minibatches = np.array_split(batch, N / M)  # create the minibatches

    return minibatches  # list of numpy.ndarray


def stopping(fun_k, jac_k, nit, max_iter, criterion):
    """
    fun_k, jac_k:
        already evaluated w.r.t. w_k
    nit:
        epochs counter
    max_iter:
        maximum number of epochs

    returns: True or False
    """

    tol = 1e-3
    stop = False

    if criterion == 0:
        stop = nit < max_iter

    if criterion == 1:
        stop = (np.linalg.norm(jac_k) > tol) and (nit < max_iter)

    if criterion == 2:
        # return (np.linalg.norm(grad_k, np.inf) > tol) and (nit < max_iter)
        stop = (np.linalg.norm(jac_k) > tol * (1 + fun_k)) and (nit < max_iter)

    return stop


def batch_grad(jac, z_t, X, y, lam, minibatch):
    """
    jac:
        callable
    z_t:
        current iterations weights
    minibatch:
        array of int32

    returns gradient w.r.t. z_t and minibatch samples
    """

    samples_x = X[minibatch, :].copy()  # matrix
    samples_y = y[minibatch].copy()     # vector

    # compute minibatch gradient
    mini_grad = jac(z_t, samples_x, samples_y, lam)

    return mini_grad


def select_step(solver, alpha, k):
    """
    solver:
        SGD-Fixed, SGD-Decreasing, SGDM, SGD-Armijo, MSL-SGDM-C, MSL-SGDM-R
    alpha:
        initial stepsize
    k:
        epochs counter

    returns: stepsize for the chosen solver
    """

    if solver in ("SGD-Fixed", "SGDM", "SGD-Armijo", "MSL-SGDM-C", "MSL-SGDM-R"):
        pass  # return initial stepsize

    elif solver == "SGD-Decreasing":
        alpha = alpha / (k + 1)

    return alpha


def select_direction(solver, beta0, grad_t, d_t):
    """
    beta0: initial momentum term
    grad_t: w.r.t. z_t
    d_t: previous iteration direction

    returns: current iteration direction
    """

    # allocate next direction
    d_next = np.empty_like(d_t)

    if solver in ("SGD-Fixed", "SGD-Decreasing", "SGDM", "SGD-Armijo"):
        # set negative gradient or damped as the direction
        d_next = -((1 - beta0) * grad_t + beta0 * d_t)

    elif solver == "MSL-SGDM-C":
        # update momentum until the direction is descent
        d_next = momentum_correction(beta0, grad_t, d_t)

    elif solver == "MSL-SGDM-R":
        # if not descent set direction to damped negative gradient
        d_next = momentum_restart(beta0, grad_t, d_t)

    return d_next


# %% utils sls


def momentum_correction(beta0, grad_t, d_t):
    """
    beta0: initial momentum term
    grad_t: w.r.t. z_t
    d_t: previous iteration direction

    returns: current iteration direction
    """

    beta = beta0  # starting momentum tern
    delta = 0.75  # momentumterm  damping factor

    d_next = -((1 - beta) * grad_t + beta * d_t)  # starting direction

    q = 0  # momentum term rejections counter
    while (not np.dot(grad_t, d_next) < 0) and (q < 20):
        beta = delta * beta  # reduce momentum term

        # update direction with reduced momentum term
        d_next = -((1 - beta) * grad_t + beta * d_t)

        q += 1

    return d_next


def momentum_restart(beta0, grad_t, d_t):
    """
    beta0: initial momentum term
    grad_t: w.r.t. z_t
    d_t: previous iteration direction

    returns: current iteration direction
    """

    d_next1 = -(1 - beta0) * grad_t
    d_next2 = -beta0 * d_t

    if np.dot(grad_t, d_next1 + d_next2) < 0:  # if descent direction
        d_next = d_next1 + d_next2

    else:
        d_next = d_next1  # restart with d=d0=0

    return d_next


def reset_step(N, alpha, alpha0, M, t):
    """
    N:
        dataset size
    alpha:
        previous iteration step-size
    alpha0:
        initial step-size
    alpha0:
        initial stepsize
    M:
        minibatch size
    t:
        iterations counter

    returns: starting stepsize for Armijo line search
    """

    opt = 2  # step-size restart rule
    a = 2    # hyperparameter

    if t == 0 or opt == 1:
        alpha = alpha0

    elif opt == 0:
        pass

    elif opt == 2:
        alpha = alpha * a**(M / N)

    return alpha


def armijo_method(z_t, d_t, X, y, lam, alpha_old, alpha_init, M, t,
                  fun, f_and_df):
    """
    z_t:
        previous iteration weights
    d_t:
        current iteration direction
    alpha_old:
        previous iteration stepsize
    alpha_init:
        initial stepsize
    M:
        minibatch size
    t:
        iterations counter
    fun, f_and_df:
        callables

    returns: selected step-size and next iteration weights
    """

    delta = 0.75  # step-size damping factor
    gamma = 0.1   # Armijo condition coefficient

    # reset step-size
    alpha = reset_step(y.size, alpha_old, alpha_init, M, t) / delta

    z_next = z_t + alpha * d_t                # model update
    fun_z, grad_z = f_and_df(z_t, X, y, lam)  # w.r.t. z_t
    fun_next = fun(z_next, X, y, lam)         # w.r.t. potential next weights

    # general Armijo condition
    condition = fun_next - (fun_z + gamma * alpha * np.dot(grad_z, d_t))

    q = 0  # step-size rejections counter
    while (not condition <= 0) and (q < 20):
        alpha = delta * alpha  # reduce step-size

        z_next = z_t + alpha * d_t         # model update
        fun_next = fun(z_next, X, y, lam)  # w.r.t. potential next weights

        # general Armijo condition
        condition = fun_next - (fun_z + gamma * alpha * np.dot(grad_z, d_t))

        q += 1

    return alpha, z_next
