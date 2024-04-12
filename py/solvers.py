# -*- coding: utf-8 -*-

# %% Packages
import time
# import sys
import numpy as np
import numpy.linalg as la
from scipy.optimize import OptimizeResult#, fsolve

# standard status messages of optimizers
_status_message = {"success": "Optimization terminated successfully.",
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   "maxiter": "Maximum number of epochs has been exceeded.",
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.',
                   "nan": "NaN result encountered.",
                   'out_of_bounds': 'The result is outside of the provided '
                                    'bounds.'}

# %% One function for all solvers

def minibatch_gd(w0, X, y, lam, M, alpha0, beta0, max_epochs, solver, stop,
                  delta_a, gamma, delta_m, fun, jac, f_and_df, **options):
    """
    Mini-batch Gradient Descent variants

    Parameters
    ----------
    w0 : numpy.ndarray
        initial guess, size p
    X : scipy.sparse.csr_matrix
        dataset, size Nxp
    y : numpy.ndarray
        response variable, size N
    lam : float
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

    # w_seq = np.empty((max_epochs + 1, p))   # weights per epoch sequence
    fun_seq = np.empty(max_epochs + 1)        # full fun per epoch sequence
    # grad_seq = np.empty_like(w_seq)         # full grad per epoch sequence
    time_seq = np.empty_like(fun_seq)         # time per epoch sequence

    w_k = np.asarray(w0).flatten()            # starting solution
    fun_k, grad_k = f_and_df(w_k, X, y, lam)  # full fun and grad w.r.t. w0

    # w_seq[0, :] = w_k                       # add starting solution
    fun_seq[0] = fun_k                        # add full fun evaluation
    # grad_seq[0, :] = grad_k                 # add full grad evaluation
    time_seq[0] = 0.                          # count from 0

    start = time.time()                       # start time counter
    warnflag = 0                              # solver status issues
    k = 0                                     # epochs counter
    while _stopping(fun_k, grad_k, k, max_epochs, criterion=stop):

        # split dataset indices randomly
        minibatches = _shuffle_dataset(y.size, k, M)

        z_t = w_k.copy()          # iterations starting model
        d_t = np.zeros_like(z_t)  # iterations starting direction
    
        # iterations starting step-size
        alpha_t = alpha0 / (k + 1.) if solver == "SGD-Decreasing" else alpha0

        # Adam iterations initialization
        m_t = np.zeros_like(z_t)  # first moment
        v_t = np.zeros_like(z_t)  # second moment

        # q_t = 0

        for t, minibatch in enumerate(minibatches):
            # t : iteration number
            # minibatch : numpy.ndarray of np.int32, minibatch indices

            # get minibatch samples selecting rows
            samples_x = X[minibatch]  # scipy.sparse.csr_matrix
            samples_y = y[minibatch]  # numpy.ndarray

            # samples grad w.r.t. current iteration solution
            grad_t = jac(z_t, samples_x, samples_y, lam)

            # --------- # select direction

            if solver in ("SGD-Fixed", "SGD-Decreasing", "SGDM", "SGD-Armijo"):
                d_t = -((1. - beta0) * grad_t + beta0 * d_t)

            elif solver == "MSL-SGDM-C":
                try:
                    # update momentum until the direction is significantly descent
                    d_t = _momentum_correction(beta0, grad_t, d_t, delta_m)
                except _MomentumCorrectionError as mce:
                    # momentum correction failed to find a descent direction
                    warnflag = 5
                    print(f"Momentum correction break at k={k} and t={t} caused by:\n", mce)
                    print("-----")
                    break

            elif solver == "MSL-SGDM-R":
                d_t = _momentum_restart(beta0, grad_t, d_t)

            elif solver in ("Adam", "Adamax", "MSL-Adam"):
                m_t, v_t, d_t = _adams(solver, m_t, v_t, grad_t, t)

            # --------- # line search

            if solver in ("SGD-Armijo", "MSL-SGDM-C", "MSL-SGDM-R", "MSL-Adam"):
                # print(f"grad*dir = {np.dot(grad_t, d_t):.6f}")
                if not _check_descent(grad_t, d_t):
                    # warnflag = 1
                    # print(f"Descent direction break at k={k} and t={t}")
                    # print("-----")
                    # break
                    raise _SearchDirectionError("Violated descent direction "
                                                f"at k={k}, t={t}, "
                                                f"g*d={grad_t.dot(d_t):.6f}")

                try:
                    alpha = _reset_step(y.size, alpha_t, alpha0, M, t)
                    # weights update with line search
                    alpha_t, _ = _armijo_method(z_t, d_t, grad_t, alpha, delta_a,
                                            gamma, fun, (samples_x, samples_y, lam))
                    # q_t += q
                except _LineSearchError as lse:
                    # line search failed to find a suitable learning rate
                    warnflag = 2
                    print(f"Line search break at k={k} and t={t} caused by:\n", lse)
                    print("-----")
                    break

            # --------- #

            z_t += alpha_t * d_t  # update iterations weights

        k += 1
        # print(q_t)

        w_k = z_t                                 # solution found
        fun_k, grad_k = f_and_df(w_k, X, y, lam)  # full fun and grad w.r.t. w_k

        # w_seq[k, :] = w_k                       # add last solution found
        fun_seq[k] = fun_k                        # add full fun evaluation
        # grad_seq[k, :] = grad_k                 # add full grad evaluation
        time_seq[k] = time.time() - start         # time per epoch

    # check errors
    # if warnflag == 2:
    #     msg = _status_message["pr_loss"]
    msg, warnflag = _check_errors(warnflag, k, max_epochs, grad_k, fun_k, w_k)

    result = OptimizeResult(fun=fun_k, x=w_k, jac=grad_k, status=warnflag,
                            success=(warnflag in (0, 3)), message=msg, nit=k,
                            solver=solver, minibatch_size=M,
                            runtime=time_seq[k], time_per_epoch=time_seq,
                            step_size=alpha0, momentum=beta0, fun_per_epoch=fun_seq)

    return result


# %% utils

def _check_errors(warnflag, k, max_epochs, grad_k, fun_k, w_k):
    if warnflag == 1:
        msg = "Violated descent direction"
    elif warnflag == 2:
        msg = "Failed line search"
    elif warnflag == 5:
        msg = "Failed momentum correction"
    elif k >= max_epochs:
        warnflag = 3
        msg = _status_message["maxiter"]
    elif np.isnan(grad_k).any() or np.isnan(fun_k) or np.isnan(w_k).any():
        warnflag = 4
        msg = _status_message["nan"]
    else:
        msg = _status_message["success"]

    return msg, warnflag


def _check_descent(grad, d):
    # 0. is the mathematical formulation
    # need to check also the magnitude
    val = grad.dot(d)  # float
    return val < -1e-6


def _shuffle_dataset(N, k, M):
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


def _stopping(fun_k, jac_k, nit, max_iter, criterion):
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

def _adams(solver, m_t, v_t, grad_t, t):
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

    beta1 = 0.9    # for vector m^t
    beta2 = 0.999  # for vector v^t
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
        if not _check_descent(grad_t, d_t):
            # initial moments
            m_0 = np.zeros_like(m_t)
            v_0 = np.zeros_like(v_t)
            # recompute moments
            m_t = beta1 * m_0 + (1. - beta1) * grad_t
            v_t = beta2 * v_0 + (1. - beta2) * np.square(grad_t)
            # bias correction
            m_tcap = m_t / (1. - beta1**(t + 1))
            v_tcap = v_t / (1. - beta2**(t + 1))
            # inverse matrix and recompute direction
            V_k = np.linalg.inv(np.diag(np.sqrt(v_tcap)) + eps * I)
            d_t = -np.dot(V_k, m_tcap)

    return m_t, v_t, d_t


# %% utils sls

def _momentum_correction(beta0, grad_t, d_t, delta):
    """
    Momentum correction procedure, dataset not required
    Damp the momentum term until the next direction is descent
    Dataset not required

    Parameters
    ----------
    beta0 : int
        initial momentum term.
    grad_t : numpy.ndarray
        samples gradient w.r.t. z_t.
    d_t : numpy.ndarray
        previous iteration direction
        if t==0 is the null direction
    delta : float
        damping factor.

    Returns
    -------
    d_next : numpy.ndarray
        current iteration direction
    """

    max_iter = 100  # momentum correction maximum iterations
    beta = beta0    # starting momentum term

    def phi(beta):
        return -((1. - beta) * grad_t + beta * d_t)

    d1 = phi(beta) # starting direction

    q = 0  # momentum term rejections counter
    while (not _check_descent(grad_t, d1)) and (q < max_iter):

        if not _check_momentum(beta):
            raise _MomentumCorrectionError("Momentum term outside range. "
                                           f"{beta:.6f}, q={q}, "
                                           f"g*d={grad_t.dot(d1):.6f}")
        # TODO: check something like if the correction is improving
        # like if it's decreasing

        # give more importance to the negative gradient
        beta *= delta   # reduce momentum term
        d1 = phi(beta)  # update direction

        q += 1

    if q >= max_iter:
        raise _MomentumCorrectionError("Maximum number of momentum " +
                                       "corrections iterations has been exceeded.")

    return d1


def _momentum_restart(beta0, grad_t, d_t):
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

    d_t = -((1. - beta0) * grad_t + beta0 * d_t)

    d_next = d_t if _check_descent(grad_t, d_t) else -(1. - beta0) * grad_t

    return d_next


def _reset_step(N, alpha, alpha0, M, t):
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
    a = 2.   # hyperparameter

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


def _armijo_method(z_t, d_t, grad_t, alpha_reset, delta, gamma, fun, args=()):
    """
    Armijo (stochastic) line search

    Parameters
    ----------
    z_t : numpy.ndarray
        previous iteration weights.
    d_t : numpy.ndarray
        current iteration (descent) direction.
    grad_t : numpy.ndarray
        current iteration mini-batch gradient norm.
    alpha_reset : float
        resetted step-size at start of the optimization.
    delta : float
        ste-size damping.
    gamma : float
        Armijo condition aggressiveness.
    fun : callable
        objective function, to be minimized.
    args : tuple, optional
        (samples_x, samples_y, lam). The default is ().
        (scipy.sparse.csr_matrix, numpy.ndarray, float)

    Raises
    ------
    _LineSearchError
        DESCRIPTION.

    Returns
    -------
    alpha : float
        accepted step-size.
    """

    max_iter = 100  # line search maximum iterations

    # set starting step-size
    # if alpha_reset < 1:
        # alpha = alpha_reset / delta
    # else:
    #     alpha = alpha_reset  # this can be high

    alpha = alpha_reset / delta
    if not alpha < 1.:
        alpha = 1.

    def phi(alpha):
        return fun(z_t + alpha * d_t, *args)

    phi0 = phi(0.)             # samples fun w.r.t. z_t
    derphi0 = grad_t.dot(d_t)  # should be a negative float
    phi1 = phi(alpha)          # fun w.r.t. new solution

    # stochastic Armijo condition
    armijo_thresh = phi0 + gamma * alpha * derphi0
    armijo_condition = phi1 - armijo_thresh

    improve_length = 4
    phi_seq = np.empty(improve_length)
    q = 0  # step-size rejections counter
    while (not armijo_condition <= 0.) and (q < max_iter):

        phi_seq[q % improve_length] = phi1

        # check if step-size is in range
        if not _check_step(alpha):
            raise _LineSearchError(f"Learning rate outside range. {alpha:.6f}"
                                   f"\ng*d={derphi0:.6f}, q={q}, "
                                   f"phi(0)={phi0:.6f}, phi(alpha)={phi1:.6f}"
                                   f"\nphi_seq={phi_seq}")

        # check if the line search is improving
        # phi(alpha0) > phi(alpha1) > phi(alpha2) > phi(alpha3)
        if (not phi_seq[0] > phi_seq[-1]) and (q % improve_length == 3):
            raise _LineSearchError("Line search not improving. "
                                   f"stop at q={q}, alpha={alpha}"
                                   f"\ng*d={derphi0:.6f}, "
                                   f"phi(0)={phi0}"
                                   f"\nphi_seq={phi_seq}")

        alpha *= delta     # reduce step-size
        phi1 = phi(alpha)  # fun w.r.t. new solution

        # stochastic Armijo condition
        armijo_thresh = phi0 + gamma * alpha * derphi0
        armijo_condition = phi1 - armijo_thresh

        q += 1

    if q >= max_iter:
        raise _LineSearchError("Maximum number of line search iterations" +
                               "has been exceeded.")

    return alpha, q


def _check_step(alpha):
    alpha_min = 0.
    alpha_max = 1.

    return alpha_min < alpha <= alpha_max


def _check_momentum(beta):
    beta_min = 0.
    beta_max = 1.

    return beta_min < beta < beta_max

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


# def check_step(alpha, d, x):
#     """
#     Check if the step-size is in the specified range

#     Parameters
#     ----------
#     alpha : float
#         step-size to check
#     d : numpy.ndarray
#         direction found at previous iteration, same size as x
#     x : numpy.ndarray
#         solution found at previous iteration

#     Returns
#     -------
#     boolean
#     """

#     # relative length of the direction
#     # check if there is a significant improvement
#     # s_max = np.max(np.absolute(d) / (np.absolute(x) + 1))

#     # alpha_min = sys.float_info.epsilon**(2/3) / s_max
#     alpha_min = sys.float_info.epsilon

#     alpha_max = 1e3 / s_max

#     return alpha_min <= alpha <= alpha_max


# def lipschitz(L, x, y, jac_x, jac_y):
#     """ Lipschitz L constant """
#     return np.linalg.norm(jac_x - jac_y) - L * np.linalg.norm(x - y)


# %% exception utils

class _SearchDirectionError(RuntimeError):
    pass

class _LineSearchError(RuntimeError):
    pass

class _MomentumCorrectionError(RuntimeError):
    pass
