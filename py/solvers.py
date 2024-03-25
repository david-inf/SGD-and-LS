# -*- coding: utf-8 -*-

# %% Packages
import time
import sys
import numpy as np
import numpy.linalg as la
from scipy.optimize import OptimizeResult#, fsolve
from joblib import Parallel, delayed

# %% One function for all solvers


# def sgd(w0, X, y, lam, M, alpha0, beta0, epochs, solver, stop,
#         fun, jac, f_and_df):
#     """
#     alpha0:
#         given learning rate
#     beta0:
#         given momentum term
#     solver:
#         SGD-Fixed, SGD-Decreasing, SGDM, SGD-Armijo, MSL-SGDM-C, MSL-SGDM-R
#     fun, jac, f_and_df: callables
#     """

#     # p = w0.size  # intercept and features size

#     # w_seq = np.empty((epochs + 1, p))  # weights sequence
#     fun_seq = np.empty(epochs + 1)       # full objective function sequence
#     # grad_seq = np.empty_like(w_seq)    # full gradient sequence

#     time_seq = np.empty_like(fun_seq)    # time per epoch sequence

#     w_k = w0.copy()                           # starting solution
#     # fun_k = fun(w_k, X, y, lam)             # full w.r.t. starting solution
#     # grad_k = jac(w_k, X, y, lam             # full w.r.t. starting solution
#     fun_k, grad_k = f_and_df(w_k, X, y, lam)  # full w.r.t. starting solution

#     # w_seq[0, :] = w_k        # add starting solution
#     fun_seq[0] = fun_k         # add full evaluation
#     # grad_seq[0, :] = grad_k  # add full evaluation

#     time_seq[0] = 0.0    # count from 0
#     start = time.time()  # start time counter

#     k = 0  # epochs counter
#     while stopping(fun_k, grad_k, k, epochs, criterion=stop):

#         # split dataset indices randomly
#         minibatches = shuffle_dataset(y.size, k, M)

#         z_t = w_k.copy()          # ietrations starting model
#         d_t = np.zeros_like(z_t)  # allocate iterations direction

#         # start every epoch with the given step-size
#         alpha_t = alpha0

#         if solver == "SGD-Decreasing":
#             # decrease stepsize at every epoch
#             alpha_t = alpha0 / (k + 1)

#         for t, minibatch in enumerate(minibatches):
#             # t: iteration number
#             # minibatch: numpy.ndarray of np.int32, contains minibatch indices

#             # get minibatch samples
#             samples_x = X[minibatch, :].copy()  # matrix
#             samples_y = y[minibatch].copy()     # vector

#             # samples w.r.t. iteration solution
#             grad_t = jac(z_t, samples_x, samples_y, lam)

#             # direction: anti-gradient or damped
#             d_t = select_direction(solver, beta0, grad_t, d_t)

#             if solver in ("SGD-Armijo", "MSL-SGDM-C", "MSL-SGDM-R"):
#                 # reset step-size
#                 alpha = reset_step(y.size, alpha_t, alpha0, M, t)

#                 # weights update with line search
#                 alpha_t = armijo_method(
#                     z_t, d_t, samples_x, samples_y, lam, alpha, fun, f_and_df)

#             # update weights
#             z_t += alpha_t * d_t

#         k += 1

#         w_k = z_t.copy()                          # solution found
#         # fun_k = fun(w_k, X, y, lam)             # full w.r.t. last solution found
#         # grad_k = jac(w_k, X, y, lam)            # full w.r.t. last solution found
#         fun_k, grad_k = f_and_df(w_k, X, y, lam)  # full w.r.t. last solution found

#         # w_seq[k, :] = w_k         # add last solution found
#         fun_seq[k] = fun_k          # add full evaluation
#         # grad_seq[k, :] = grad_k   # add full evaluation

#         time_seq[k] = time.time() - start  # time per epoch

#     result = OptimizeResult(fun=fun_k.copy(), x=w_k.copy(), jac=grad_k.copy(),
#                             success=(k > 1), solver=solver, minibatch_size=M,
#                             nit=k, runtime=time_seq[k], time_per_epoch=time_seq,
#                             step_size=alpha0, momentum=beta0,
#                             fun_per_epoch=fun_seq)

#     return result


def minibatch_gd(w0, X, y, lam, M, alpha0, beta0, epochs, solver, stop,
                 fun, jac, f_and_df):
    """
    Mini-batch Gradient Descent variants
    Handles X as CSR matrix

    Parameters
    ----------
    w0 : numpy.ndarray
        initial guess, size p
    X : scipy.sparse.csr_matrix
        dataset, size Nxp
    y : numpy.ndarray
        response variable, size N
    lam : int
        regularization term
    M : int
        minibatches size
    alpha0 : float
        given learning rate
    beta0 : float
        given momentum term
    epochs : int
        maximum number of epochs
    solver : string
        SGD-Fixed, SGD-Decreasing, SGDM, SGD-Armijo, MSL-SGDM-C, MSL-SGDM-R
    stop : int
        stopping criterion to be used
    fun, jac, f_and_df : callables
        objective function, gradient, both as a tuple

    Returns
    -------
    OptimizeResult
    """

    # w_seq = np.empty((epochs + 1, p))  # weights sequence
    fun_seq = np.empty(epochs + 1)       # full objective function sequence
    # grad_seq = np.empty_like(w_seq)    # full gradient sequence

    time_seq = np.empty_like(fun_seq)    # time per epoch sequence

    w_k = np.asarray(w0).flatten()            # starting solution
    # fun_k = fun(w_k, X, y, lam)             # full w.r.t. starting solution
    # grad_k = jac(w_k, X, y, lam)            # full w.r.t. starting solution
    fun_k, grad_k = f_and_df(w_k, X, y, lam)  # full w.r.t. starting solution

    # w_seq[0, :] = w_k        # add starting solution
    fun_seq[0] = fun_k         # add full evaluation
    # grad_seq[0, :] = grad_k  # add full evaluation

    time_seq[0] = 0.0    # count from 0
    start = time.time()  # start time counter

    k = 0  # epochs counter
    while stopping(fun_k, grad_k, k, epochs, criterion=stop):

        # split dataset indices randomly
        minibatches = shuffle_dataset(y.size, k, M)

        z_t = w_k.copy()          # iterations starting model
        d_t = np.zeros_like(z_t)  # allocate iterations direction

        # start every epoch with the given step-size
        alpha_t = alpha0

        if solver == "SGD-Decreasing":
            # decrease stepsize at every epoch
            alpha_t = alpha0 / (k + 1)

        for t, minibatch in enumerate(minibatches):
            # t : iteration number
            # minibatch : numpy.ndarray, minibatch indices

            # get minibatch samples
            samples_x = X[minibatch]  # scipy.sparse.csr_matrix
            samples_y = y[minibatch]  # numpy.ndarray

            # samples w.r.t. iteration solution
            grad_t = jac(z_t, samples_x, samples_y, lam)

            # direction: anti-gradient or damped
            d_t = select_direction(solver, beta0, grad_t, d_t)

            if solver in ("SGD-Armijo", "MSL-SGDM-C", "MSL-SGDM-R"):
                # reset step-size
                alpha = reset_step(y.size, alpha_t, alpha0, M, t)

                # weights update with line search
                alpha_t = armijo_method(
                    z_t, d_t, samples_x, samples_y, lam, alpha, fun, f_and_df)

            # update weights
            z_t += alpha_t * d_t

        k += 1

        w_k = z_t.copy()                          # solution found
        # fun_k = fun(w_k, X, y, lam)             # full w.r.t. last solution found
        # grad_k = jac(w_k, X, y, lam)            # full w.r.t. last solution found
        fun_k, grad_k = f_and_df(w_k, X, y, lam)  # full w.r.t. last solution found

        # w_seq[k, :] = w_k         # add last solution found
        fun_seq[k] = fun_k          # add full evaluation
        # grad_seq[k, :] = grad_k   # add full evaluation

        time_seq[k] = time.time() - start  # time per epoch

    result = OptimizeResult(fun=fun_k, x=w_k, jac=grad_k,
                            success=(k > 1), solver=solver, minibatch_size=M,
                            nit=k, runtime=time_seq[k], time_per_epoch=time_seq,
                            step_size=alpha0, momentum=beta0,
                            fun_per_epoch=fun_seq)

    return result


# %% Prova


def minibatch_gd_parallel(w0, X, y, lam, M, alpha0, beta0, epochs, solver, stop,
                          fun, jac, f_and_df):
    """
    Mini-batch Gradient Descent variants
    Handles X as CSR matrix

    Parameters
    ----------
    w0 : numpy.ndarray
        initial guess, size p
    X : scipy.sparse.csr_matrix
        dataset, size Nxp
    y : numpy.ndarray
        response variable, size N
    lam : int
        regularization term
    M : int
        minibatches size
    alpha0 : float
        given learning rate
    beta0 : float
        given momentum term
    epochs : int
        maximum number of epochs
    solver : string
        SGD-Fixed, SGD-Decreasing, SGDM, SGD-Armijo, MSL-SGDM-C, MSL-SGDM-R
    stop : int
        stopping criterion to be used
    fun, jac, f_and_df : callables
        objective function, gradient, both as a tuple

    Returns
    -------
    OptimizeResult
    """

    fun_seq = np.empty(epochs + 1)       # full objective function sequence
    time_seq = np.empty_like(fun_seq)    # time per epoch sequence

    w_k = np.asarray(w0).flatten()            # starting solution
    fun_k, grad_k = f_and_df(w_k, X, y, lam)  # full w.r.t. starting solution

    fun_seq[0] = fun_k         # add full evaluation
    time_seq[0] = 0.0          # count from 0
    start = time.time()        # start time counter

    k = 0  # epochs counter
    while stopping(fun_k, grad_k, k, epochs, criterion=stop):

        minibatches = shuffle_dataset(y.size, k, M)

        z_t = w_k.copy()          # iterations starting model
        d_t = np.zeros_like(z_t)  # allocate iterations direction
        alpha_t = alpha0

        if solver == "SGD-Decreasing":
            alpha_t = alpha0 / (k + 1)

        grad_t_list = Parallel(n_jobs=-1)(
            delayed(jac)(z_t, X[minibatch], y[minibatch], lam) for minibatch in minibatches
        )

        for grad_t in grad_t_list:
            d_t = select_direction(solver, beta0, grad_t, d_t)
            z_t += alpha_t * d_t

        k += 1

        w_k = z_t.copy()                          # solution found
        fun_k, grad_k = f_and_df(w_k, X, y, lam)  # full w.r.t. last solution found

        fun_seq[k] = fun_k          # add full evaluation
        time_seq[k] = time.time() - start  # time per epoch

    result = OptimizeResult(fun=fun_k, x=w_k, jac=grad_k,
                            success=(k > 1), solver=solver, minibatch_size=M,
                            nit=k, runtime=time_seq[k], time_per_epoch=time_seq,
                            step_size=alpha0, momentum=beta0,
                            fun_per_epoch=fun_seq)

    return result


# %% utils


def shuffle_dataset(N, k, M):
    """
    Shuffle dataset indices, dataset not required

    Parameters
    ----------
    N : int
        dataset size
    k : int
        epochs counter
    M : int
        minibatch size

    Returns
    -------
    minibatches : list of numpy.ndarray of numpy.int32
    """

    batch = np.arange(N)  # dataset indices

    rng = np.random.default_rng(k)  # set different seed every epoch
    rng.shuffle(batch)              # shuffle indices

    # array_split is expensive, consider another strategy
    minibatches = np.array_split(batch, N / M)  # create the minibatches

    return minibatches


def stopping(fun_k, jac_k, nit, max_iter, criterion):
    """
    Selection of stopping criterion, dataset not required

    Parameters
    ----------
    fun_k, jac_k : numpy.ndarray
        already evaluated w.r.t. w_k
    nit : int
        epochs counter
    max_iter : int
        maximum number of epochs
    criterion : int
        stopping criterion to be used

    Returns
    -------
    stop : boolean
    """

    tol = 1e-3
    stop = False

    if criterion == 0:
        stop = nit < max_iter

    elif criterion == 1:
        stop = (la.norm(jac_k) > tol) and (nit < max_iter)

    elif criterion == 2:
        # return (np.linalg.norm(grad_k, np.inf) > tol) and (nit < max_iter)
        stop = (la.norm(jac_k) > tol * (1 + fun_k)) and (nit < max_iter)

    return stop


# def batch_grad(jac, z_t, X, y, lam, batch_idx):
#     """
#     jac: callable
#     z_t:
#         current iterations weights
#     batch_idx: array of np.int32
#         minibatch indices

#     returns gradient w.r.t. z_t and minibatch samples
#     """

#     # get minibatch samples
#     samples_x = X[batch_idx, :].copy()  # matrix
#     samples_y = y[batch_idx].copy()     # vector

#     # compute minibatch gradient
#     mini_grad = jac(z_t, samples_x, samples_y, lam)

#     return mini_grad


# def select_step(solver, alpha, k):
#     """
#     solver:
#         SGD-Fixed, SGD-Decreasing, SGDM, SGD-Armijo, MSL-SGDM-C, MSL-SGDM-R
#     alpha:
#         initial stepsize
#     k:
#         epochs counter

#     returns: stepsize for the chosen solver
#     """

#     if solver == "SGD-Decreasing":
#         # decrease stepsize on every epoch
#         alpha = alpha / (k + 1)

#     return alpha


def select_direction(solver, beta0, grad_t, d_t):
    """
    Select direction based on the selected solver
    Dataset not required for this operation

    Parameters
    ----------
    solver : string
    beta0 : int
        initial momentum term
    grad_t : numpy.ndarray of size d_t.size
        w.r.t. z_t
    d_t : numpy.ndarray
        previous iteration direction

    Returns
    -------
    d_next : numpy.ndarray
        current iteration direction
    """

    # allocate next direction
    d_next = np.empty_like(d_t)

    # beta = beta0

    if solver in ("SGD-Fixed", "SGD-Decreasing", "SGD-Armijo"):
        # set negative gradient
        d_next = -grad_t

    elif solver == "SGDM":
        # damped negative gradient
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
    Momentum correction procedure, dataset not required
    Damp the momentum term until the next direction is descent
    Dataset not required

    Parameters
    ----------
    beta0 : int
        initial momentum term
    grad_t : numpy.ndarray
        w.r.t. z_t
    d_t : numpy.ndarray
        previous iteration direction
        if t==0 every element is 0

    Returns
    -------
    d_next : numpy.ndarray
        current iteration direction
    """

    beta = beta0  # starting momentum tern
    delta = 0.5  # momentum term  damping factor

    d_next = -((1 - beta) * grad_t + beta * d_t)  # starting direction
    # bestbeta = beta0

    q = 0  # momentum term rejections counter
    while (not np.dot(grad_t, d_next) < 0) and (q < 15):

        # give more importance to the negative gradient
        beta *= delta  # reduce momentum term

        # update direction with reduced momentum term
        d_next = -((1 - beta) * grad_t + beta * d_t)

        q += 1

    return d_next


def momentum_restart(beta0, grad_t, d_t):
    """
    Momentum restart procedure, dataset not required
    Check if the next direction is descent
    Dataset not required

    Parameters
    ----------
    beta0 : int
        initial momentum term
    grad_t: numpy.ndarray
        w.r.t. z_t
    d_t: numpy.ndarray
        previous iteration direction

    Returns
    -------
    d_next : numpy.ndarray
        current iteration direction
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
    Step-size resetting procedure
    Dataset not required

    Parameters
    ----------
    N : int
        dataset size
    alpha : float
        previous iteration step-size
    alpha0 : float
        initial step-size
    M : int
        minibatch size
    t : int
        iterations counter

    Returns
    -------
    alpha : float
        starting stepsize for Armijo line search
    """

    opt = 2  # step-size restart rule
    a = 2    # hyperparameter

    if t == 0 or opt == 1:
        # set given learning rate
        alpha = alpha0

    elif opt == 0:
        # set previous iteration step-size
        pass

    elif opt == 2:
        # a little higher than the previous
        alpha *= a**(M / N)

    return alpha


def armijo_method(z_t, d_t, samples_x, samples_y, lam, alpha_reset, fun, f_and_df):
    """
    Armijo (stochastic) line search
    Required dataset (current minibatch samples)
    for evaluating the Armijo condition

    Parameters
    ----------
    z_t : numpy.ndarray
        previous iteration weights
    d_t : numpy.ndarray
        current iteration (descent) direction
    samples_x : scipy.sparse.csr_matrix
        samples selected from CSR matrix X
    samples_y : numpy.ndarray
        samples selected from array y
    lam : int
        regularization term
    alpha_reset : float
        step-size resetted from previous iteration
    N : int
        dataset size
    fun, f_and_df : callables
        objective function, obj and gradient as a tuple

    Returns
    -------
    alpha : float
        selected step-size
    """

    delta = 0.9  # step-size damping factor
    gamma = 0.01  # Armijo condition coefficient

    # set starting step-size
    if alpha_reset < 1:
        alpha = alpha_reset / delta

    else:
        alpha = alpha_reset

    # samples w.r.t. z_t
    fun_t, grad_t = f_and_df(z_t, samples_x, samples_y, lam)

    z_next = z_t + alpha * d_t  # model update

    # samples w.r.t. potential next weights
    fun_next = fun(z_next, samples_x, samples_y, lam)
    # fun_next, grad_next = f_and_df(z_next, samples_x, samples_y, lam)

    # stochastic Armijo condition
    # armijo_thresh should be greater than fun_next
    armijo_thresh = fun_t + gamma * alpha * np.dot(grad_t, d_t)
    armijo_condition = fun_next - armijo_thresh

    q = 0  # step-size rejections counter
    while not armijo_condition <= 0 and q < 100:

        # step_in_range = check_step(alpha, gamma, d_t, z_t, z_next, grad_t, grad_next)
        if not check_step(alpha, d_t, z_t):
            return alpha / delta

        alpha *= delta         # reduce step-size
        z_next += alpha * d_t  # model update

        # samples w.r.t. potential next weights
        fun_next = fun(z_next, samples_x, samples_y, lam)

        # general Armijo condition
        armijo_thresh = fun_t + gamma * alpha * np.dot(grad_t, d_t)
        armijo_condition = fun_next - armijo_thresh

        q += 1

    return alpha


# def check_step(alpha, gamma, d, x, y, jac_x, jac_y):
#     """
#     check if the step-size is in the specified range
#     """
#     # relative length of the direction
#     # check if there is a significant improvement
#     s_max = np.max(np.divide(d, x + 1))

#     alpha_min = sys.float_info.epsilon**(2/3) / s_max
#     # alpha_min = 2 * (1 - gamma) / fsolve(lipschitz, 1, args=(x, y, jac_x, jac_y)[0])

#     alpha_max = 1e3 / s_max

#     return alpha_min <= alpha <= alpha_max


def check_step(alpha, d, x):
    """
    Check if the step-size is in the specified range

    Parameters
    ----------
    alpha : float
        step-size to check
    d : numpy.ndarray
        direction found at previous iteration
    x : numpy.ndarray
        solution found at previous iteration

    Returns
    -------
    boolean
    """

    # relative length of the direction
    # check if there is a significant improvement
    s_max = np.max(np.divide(d, x + 1))

    alpha_min = sys.float_info.epsilon**(2/3) / s_max

    alpha_max = 1e3 / s_max

    return alpha_min <= alpha <= alpha_max


def lipschitz(L, x, y, jac_x, jac_y):
    """ Lipschitz L constant """
    return np.linalg.norm(jac_x - jac_y) - L * np.linalg.norm(x - y)


# %%


# def grid_search():
    # rates = [1, 0.5, 0.25, 0.1, 0.05, 0.25, 0.1]
