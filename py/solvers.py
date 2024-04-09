# -*- coding: utf-8 -*-

# %% Packages
import time
import sys
import numpy as np
import numpy.linalg as la
from scipy.optimize import OptimizeResult#, fsolve

# %% One function for all solvers

def minibatch_gd(w0, X, y, lam, M, alpha0, beta0, epochs, solver, stop,
                  delta_a, gamma, delta_m, fun, jac, f_and_df, **options):
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
        SGD-Fixed, SGD-Decreasing, SGDM, SGD-Armijo, MSL-SGDM-C, MSL-SGDM-R,
        Adam, MSL-Adam, Adamax
    stop : int
        stopping criterion to be used
    delta_a : float
        armijo damping parameter
    gamma : float
        armijo condition parameter
    delta_m : float
        momentum correction damping parameter
    fun, jac, f_and_df : callables
        objective function, gradient, both as a tuple

    Returns
    -------
    OptimizeResult
    """

    # w_seq = np.empty((epochs + 1, p))  # weights sequence
    fun_seq = np.empty(epochs + 1)       # full objective function sequence
    # grad_seq = np.empty_like(w_seq)    # full gradient sequence

    time_seq = np.empty_like(fun_seq)  # time per epoch sequence

    w_k = np.asarray(w0).flatten()            # starting solution
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
        d_t = np.zeros_like(z_t)  # initialize iterations direction

        # start iterations with the given step-size
        # all solvers except SGD-Decreasing
        alpha_t = alpha0

        # Adam initialization
        m_t = np.zeros_like(z_t)
        v_t = np.zeros_like(z_t)

        if solver == "SGD-Decreasing":
            # decrease stepsize at every epoch
            alpha_t = alpha0 / (k + 1)

        for t, minibatch in enumerate(minibatches):
            # t : iteration number
            # minibatch : numpy.ndarray of np.int32, minibatch indices

            # get minibatch samples
            samples_x = X[minibatch]  # scipy.sparse.csr_matrix, select rows
            samples_y = y[minibatch]  # numpy.ndarray

            # samples gradient w.r.t. iteration solution
            grad_t = jac(z_t, samples_x, samples_y, lam)

            # --------- #

            # direction: anti-gradient or damped
            if solver in ("SGD-Fixed", "SGD-Decreasing", "SGDM", "SGD-Armijo"):
                # negative gradient damped or not
                d_t = -((1 - beta0) * grad_t + beta0 * d_t)

            elif solver == "MSL-SGDM-C":
                # update momentum until the direction is descent
                d_t = momentum_correction(beta0, grad_t, d_t, delta_m)

            elif solver == "MSL-SGDM-R":
                # if not descent set direction to damped negative gradient
                d_t = momentum_restart(beta0, grad_t, d_t)

            elif solver in ("Adam", "Adamax", "MSL-Adam"):
                # get vectors m^t and v^t, and direction
                m_t, v_t, d_t = adam_things(solver, m_t, v_t, grad_t, t)

            # elif solver == "MSL-Adam":
                # restart direction if not descent
                # m_t, v_t, d_t = adam_restart(solver, m_t, v_t, grad_t, t)

            # --------- #

            if solver in ("SGD-Armijo", "MSL-SGDM-C", "MSL-SGDM-R", "MSL-Adam"):
                # reset step-size
                alpha = reset_step(y.size, alpha_t, alpha0, M, t)

                # weights update with line search
                alpha_t = armijo_method(
                    z_t, d_t, samples_x, samples_y, lam, alpha, delta_a, gamma,
                    grad_t, fun)

            # update weights
            z_t += alpha_t * d_t

        k += 1

        w_k = z_t.copy()                          # solution found
        fun_k, grad_k = f_and_df(w_k, X, y, lam)  # full w.r.t. last solution found

        # w_seq[k, :] = w_k         # add last solution found
        fun_seq[k] = fun_k          # add full evaluation
        # grad_seq[k, :] = grad_k   # add full evaluation

        time_seq[k] = time.time() - start  # time per epoch

    result = OptimizeResult(fun=fun_k, x=w_k, jac=grad_k, success=(k > 1),
                            solver=solver, minibatch_size=M, nit=k,
                            runtime=time_seq[k], time_per_epoch=time_seq,
                            step_size=alpha0, momentum=beta0, fun_per_epoch=fun_seq)

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

# %% Adam

def adam_things(solver, m_t, v_t, grad_t, t):
    """
    Parameters
    ----------
    solver : string
        Adam, MSL-Adam.
    m_t : numpy.ndarray
        first moment form previous iteration.
    v_t : numpy.ndarray
        second moment from previous iteration.
    grad_t : numpy.ndarray
        mini-batch gradient.
    t : int
        iterations counter.

    Returns
    -------
    m_t : numpy.ndarray
        updated first moment.
    v_t : numpy.ndarray
        updatede second moment.
    d_t : numpy.ndarray
        new direction.
    """

    beta1 = 0.9      # for vector m^t
    beta2 = 0.999    # for vector v^t
    eps = 1e-8
    I = np.eye(v_t.size)

    # vector m^t and v^t
    m_t = beta1 * m_t + (1 - beta1) * grad_t
    v_t = beta2 * v_t + (1 - beta2) * np.square(grad_t)

    # bias correction
    m_tcap = m_t / (1 - beta1**(t + 1))
    v_tcap = v_t / (1 - beta2**(t + 1))

    # TODO: Adamax routine

    # matrix inversion and direction computation
    V_k = np.linalg.inv(np.diag(np.sqrt(v_tcap)) + eps * I)
    d_t = -np.dot(V_k, m_tcap)

    # MSL-Adam restart routine
    if solver == "MSL-Adam":
        if not np.dot(grad_t, d_t) < 0:
            # initial moments
            m_0 = np.zeros_like(m_t)
            v_0 = np.zeros_like(v_t)
            # recompute moments
            m_t = beta1 * m_0 + (1 - beta1) * grad_t
            v_t = beta2 * v_0 + (1 - beta2) * np.square(grad_t)
            # bias correction
            m_tcap = m_t / (1 - beta1**(t + 1))
            v_tcap = v_t / (1 - beta2**(t + 1))
            # inverse matrix and recompute direction
            V_k = np.linalg.inv(np.diag(np.sqrt(v_tcap)) + eps * I)
            d_t = -np.dot(V_k, m_tcap)

    return m_t, v_t, d_t


# def adam_restart(solver, m_t, v_t, grad_t, t):
#     beta1 = 0.9      # for vector m^t
#     beta2 = 0.999    # for vector v^t
#     eps = 1e-8
#     I = np.eye(v_t.size)

#     # vector m^t and v^t
#     m_t = beta1 * m_t + (1 - beta1) * grad_t
#     v_t = beta2 * v_t + (1 - beta2) * np.square(grad_t)

#     # bias correction
#     m_tcap = m_t / (1 - beta1**(t + 1))
#     v_tcap = v_t / (1 - beta2**(t + 1))

#     # matrix inversion
#     V_k = np.linalg.inv(np.diag(np.sqrt(v_tcap)) + eps * I)
#     d_t = -np.dot(V_k, m_tcap)

#     # check descent direction
#     if not np.dot(grad_t, d_t) < 0:
#         # initial moments
#         m_0 = np.zeros_like(m_t)
#         v_0 = np.zeros_like(v_t)
#         # recompute moments
#         m_t = beta1 * m_0 + (1 - beta1) * grad_t
#         v_t = beta2 * v_0 + (1 - beta2) * np.square(grad_t)
#         # bias correction
#         m_tcap = m_t / (1 - beta1**(t + 1))
#         v_tcap = v_t / (1 - beta2**(t + 1))
#         # recompute direction
#         V_k = np.linalg.inv(np.diag(np.sqrt(v_tcap)) + eps * I)
#         d_t = -np.dot(V_k, m_tcap)

#     return m_t, v_t, d_t

# %% utils sls

def momentum_correction(beta0, grad_t, d_t, delta):
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
    delta : float

    Returns
    -------
    d_next : numpy.ndarray
        current iteration direction
    """

    beta = beta0  # starting momentum tern

    d_next = -((1 - beta) * grad_t + beta * d_t)  # starting direction
    # bestbeta = beta0

    q = 0  # momentum term rejections counter
    while (not np.dot(grad_t, d_next) < 0) and (q < 100):

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


def armijo_method(z_t, d_t, samples_x, samples_y, lam, alpha_reset, delta, gamma,
                  grad_t, fun):
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
    delta : float
    gamma : float
    fun, f_and_df : callables
        objective function, obj and gradient as a tuple

    Returns
    -------
    alpha : float
        selected step-size
    """

    # set starting step-size
    if alpha_reset < 1:
        alpha = alpha_reset / delta

    else:
        alpha = alpha_reset

    # samples w.r.t. z_t
    fun_t = fun(z_t, samples_x, samples_y, lam)

    # model update
    z_next = z_t + alpha * d_t
    # samples w.r.t. potential next weights
    fun_next = fun(z_next, samples_x, samples_y, lam)

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
