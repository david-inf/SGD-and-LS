# -*- coding: utf-8 -*-

import time
import sys
import numpy as np
import numpy.linalg as la
from scipy.optimize import OptimizeResult#, fsolve
from scipy.optimize import line_search  # strong wolfe conditions

from solvers_utils import _check_errors, _check_descent, _shuffle_dataset, _stopping
from load_datasets import dataset_distrib

# %% One function for all solvers

def minibatch_gd(fun, w0, fk_args, solver, jac, f_and_df, batch_size, alpha0, beta0,
                 maxepochs, stop, delta_a, gamma, delta_m, options=None):
    """
    Stochastic Gradient Descent, minimization of a finite-sum problem.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.

            ``fun(w, *args) -> float``

        where ``w`` is a 1-D array with shape (p+1,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    w0 : array_like, shape (p+1,)
        Initial guess.
    fk_args : tuple
        Extra arguments passed to the objective function and its
        derivatives (`fun`, `jac` and `hess` functions).
    solver : str
        Type of solver. Should be one of SGD-Fixed, SGD-Decreasing, SGDM,
        SGD-Armijo, MSL-SGDM-C, MSL-SGDM-R, Adam, MSL-Adam, Adamax
    jac : callable
        Objective function gradient.
    f_and_df : callable
        Both objective function and gradient as a tuple.
    batch_size : int
        Mini-batch size.
    alpha0 : float
        Initial learning rate.
    beta0 : float
        Initial momentum term.
    maxepochs : int
        Maximum number of epochs.
    stop : int
        Stopping criterion type.
    delta_a : float
        Learning rate damping factor in line search.
    gamma : float
        Stochastic Armijo condition aggressiveness.
    delta_m : float
        Momentum term damping factor in momentum correction.

    Returns
    -------
    OptimizeResult

    """

    X, y, lam = fk_args  # full data, full respose, regul coeff

    fun_seq = np.zeros(maxepochs + 1)  # full fun per epoch sequence
    time_seq = np.zeros_like(fun_seq)  # time per epoch sequence

    wk = np.asarray(w0).flatten()      # starting solution, copies w0
    fk, gfk = f_and_df(wk, *fk_args)   # full fun and grad w.r.t. w0

    fun_seq[0] = fk                    # add full fun evaluation
    time_seq[0] = 0.                   # count from 0
    _start = time.time()                # start time counter
    warnflag = 0                       # solver status issues

    _rng = np.random.default_rng(42)  # random generator for shuffling dataset
    k = 0  # epochs counter
    _iter_ls = 0   # line search iterations counter
    _iter_msl = 0  # momentum corrections/restarts counter
    while _stopping(fk, gfk, k, maxepochs, criterion=stop):

        # split dataset randomly
        minibatches = _shuffle_dataset(y.size, batch_size, _rng)
        # if k < 2:
            # for batch in minibatches:
                # print(dataset_distrib(y[batch]))
            # print("----")
        # init iterations
        zt = wk.copy()          # starting weights
        dt = np.zeros_like(zt)  # init direction
        # select starting learning rate
        alphat = alpha0 / (k + 1.) if solver == "SGD-Decreasing" else alpha0

        # Adam initialization
        # mt = np.zeros_like(zt)
        # vt = np.zeros_like(zt)

        for t, minibatch in enumerate(minibatches):
            # t : iteration number
            # minibatch : numpy.ndarray of np.int32, minibatch indices

            # samples data, samples response, coeff
            ft_args = (X[minibatch], y[minibatch], lam)
            gft = jac(zt, *ft_args)  # gradient w.r.t. iteration weights

            # --------- # select direction

            if solver in ("SGD-Fixed", "SGD-Decreasing", "SGDM", "SGD-Armijo"):
                # when momentum term is added, direction may not be descent
                # due to unrelated mini-batches
                dt = -((1. - beta0) * gft + beta0 * dt)

            elif solver in ("MSL-SGDM-C", "MSL-SGDM-R"):
                try:
                    dt, _q_msl = _stochastic_momentum(solver, k, t, beta0, gft, dt, delta_m)
                    _iter_msl += _q_msl
                except _MomentumCorrectionError as mce:
                    warnflag = 1
                    print(f"Momentum correction break at k={k} and t={t}\n", mce)
                    print("-----")

            # elif solver in ("Adam", "Adamax", "MSL-Adam"):
            #     mt, vt, dt = _adams(solver, mt, vt, gft, t)

            # check on direction
            # if not gft.dot(dt) < 0:
            #     raise _SearchDirectionError("Violated descent direction "
            #                                 f"at k={k}, t={t}, "
            #                                 f"g*d={gft.dot(dt):.6f}")

            # --------- # line search

            if solver in ("SGD-Armijo", "MSL-SGDM-C", "MSL-SGDM-R", "MSL-Adam"):
                try:
                    alphat, _q_ls = _stochastic_armijo(
                        y.size, k, t, alphat, alpha0, zt, dt, gft, delta_a, gamma,
                        fun, ft_args)
                    _iter_ls += _q_ls
                except _SearchDirectionError as sde:
                    print(sde)
                    print("-----")
                except _LineSearchError as lse:
                    print(f"Line search break at k={k} and t={t}.\n", lse)
                    print("-----")

                # alpha_t = line_search(fun, jac, zt, dt, gft, *ft_args)[0]

            # --------- #

            zt += alphat * dt  # update iterations weights

        k += 1

        wk = zt                            # solution found
        fk, gfk = f_and_df(wk, *fk_args)   # full fun and grad w.r.t. w_k

        fun_seq[k] = fk                    # add full fun evaluation
        time_seq[k] = time.time() - _start  # time per epoch

    msg, warnflag = _check_errors(warnflag, k, maxepochs, gfk, fk, wk)

    result = OptimizeResult(fun=fk, x=wk, jac=gfk, status=warnflag, solver=solver,
                            success=(warnflag in (0, 2)), message=msg, nit=k,
                            minibatch_size=batch_size, ls_per_epoch = _iter_ls / k,
                            msl_per_epoch = _iter_msl / k,
                            runtime=time_seq[k], time_per_epoch=time_seq,
                            step_size=alpha0, momentum=beta0, fun_per_epoch=fun_seq)

    return result



def batch_gd(fun, w0, fk_args, solver, jac, alpha0, maxepochs, stop):

    fun_seq = np.empty(maxepochs + 1)  # full fun per epoch sequence
    time_seq = np.empty_like(fun_seq)  # time per epoch sequence
    sk_seq = np.empty(maxepochs)

    wk = np.asarray(w0).flatten()      # starting solution, copies w0
    fk, gfk = fun(wk, *fk_args), jac(wk, *fk_args)   # full fun and grad w.r.t. w0

    fun_seq[0] = fk                    # add full fun evaluation
    time_seq[0] = 0.                   # count from 0
    start = time.time()                # start time counter
    warnflag = 0                       # solver status issues

    k = 0  # epochs counter
    while _stopping(fk, gfk, k, maxepochs, stop):

        gfk = jac(wk, *fk_args)
        dk = -gfk

        # check on direction
        # if not gft.dot(dt) < 0:
        #     raise _SearchDirectionError("Violated descent direction "
        #                                 f"at k={k}, t={t}, "
        #                                 f"g*d={gft.dot(dt):.6f}")

        sk = alpha0 * dk
        wk += sk  # update iterations weights
        fk, gfk = fun(wk, *fk_args), jac(wk, *fk_args)  # full fun and grad w.r.t. w_k

        k += 1

        fun_seq[k] = fk                    # add full fun evaluation
        time_seq[k] = time.time() - start  # time per epoch
        sk_seq[k-1] = la.norm(sk)

    msg, warnflag = _check_errors(warnflag, k, maxepochs, gfk, fk, wk)

    result = OptimizeResult(fun=fk, x=wk, jac=gfk, status=warnflag, solver=solver,
                            success=(warnflag in (0, 2)), message=msg, nit=k,
                            minibatch_size=fk_args[1].size, ls_per_epoch = 0,
                            msl_per_epoch = 0,
                            runtime=time_seq[k], time_per_epoch=np.nan,
                            step_size=alpha0, momentum=0., fun_per_epoch=fun_seq,
                            sk_per_epoch=sk_seq)

    return result


# %% SLS

def _reset_alpha(N, alpha_old, alpha0, M, t):
    """
    Returns
    -------
    Resetted learning rate for Armijo stochastic line search.
    """

    opt = 2  # step-size restart rule
    a = 2.   # hyperparameter

    if t == 0 or opt == 1:
        # set given learning rate
        alpha_old = alpha0

    elif opt == 0:
        # set previous iteration step-size
        pass

    elif opt == 2:
        # a little higher than the previous one
        alpha_old *= a**(M / N)

    return alpha_old


def _line_search_armijo(fun, xk, dk, gfk, fk, args=(), c1=1e-4, delta=0.5, alpha0=1.):
    """Minimize over alpha, the function ``fun(xk+alpha*dk)``.

    Parameters
    ----------
    fun : callable
        Function to be minimized.
    xk : array_like
        Current point.
    dk : array_like
        Search direction.
    gfk : array_like
        Gradient of `fun` at point `xk`.
    fk : float
        Value of `f` at point `xk`.
    args : tuple, optional
        Function optional arguments. The default is ().
    c1 : float, optional
        Armijo condition aggressiveness. The default is 1e-4.
    delta : float, optional
        Step damping factor. The default is 0.5.
    alpha0 : float, optional
        Value of `alpha` at start of the optimization. The default is 1..

    Raises
    ------
    _LineSearchError

    Returns
    -------
    alpha

    """

    maxiter = 100    # line search maximum iterations
    alpha1 = alpha0  # starting step-size

    def phi(alpha):
        return fun(xk + alpha * dk, *args)

    if fk is None:
        phi0 = phi(0.)     # current function value
    else:
        phi0 = fk
    derphi0 = gfk.dot(dk)  # should be a negative float
    phi1 = phi(alpha1)     # function next value

    # check Armijo condition
    if phi1 <= phi0 + c1*alpha1*derphi0:
        return alpha1, 0

    improve_length = 3  # number of iterations to check
    phi_seq = np.empty(improve_length)

    q = 0  # step-size rejections counter
    while _check_step(alpha1):

        q += 1

        if q >= maxiter:
            msg = "Maximum number of line search iterations has been exceeded. q={q}"
            raise _LineSearchError(msg)

        phi_seq[(q - 1) % improve_length] = phi1  # add function on next point

        # check if the line search is improving
        # phi(alpha0) > phi(alpha1) > phi(alpha2) > phi(alpha3)
        if (not phi_seq[0] > phi_seq[-1]) and ((q - 1) % improve_length == improve_length - 1):
            msg = f"Line search not improving. Stop at q={q}, alpha={alpha1:.6f}" \
                  f"\ng*d={derphi0:.6f}, phi(0)={phi0:.6f}" \
                  f"\nphi_seq={phi_seq}"
            raise _LineSearchError(msg)

        alpha1 *= delta     # reduce step-size
        phi1 = phi(alpha1)  # fun w.r.t. new solution

        # check Armijo condition
        if phi1 <= phi0 + c1*alpha1*derphi0:
            return alpha1, q

    # Failed to find a suitable step length
    return None, q


def _stochastic_armijo(N, k, t, alpha_old, alpha0, zt, dt, gft, delta, c1,
                      fun, args):
    """
    Stochastic Armijo line search

    Parameters
    ----------
    N : int
        Dataset size.
    t : int
        Current iteration.
    alpha_old : float
        Previous iteration learning rate.
    alpha0 : float
        Initiale learning rate.
    zt : array_like
        Weights to update.
    dt : array_like
        Direction for weights update.
    gft : array_like
        Current mini-batch gradient.
    delta : float
        Learning rate damping.
    c1 : float
        Armijo condition aggressiveness.
    fun : callable
        Objective function.
    args : tuple
        Objective function arguments.

    Returns
    -------
    alpha_opt : float
        Accepted learning rate.
    """

    # double check on direction
    if not _check_descent(gft, dt):
        msg = "Violated descent direction at k={k}, t={t}, g*d={gft.dot(dt):.6f}"
        raise _SearchDirectionError(msg)

    # check learning rate upper-bound
    M = args[1].size
    alpha_init = _reset_alpha(N, alpha_old, alpha0, M, t) / delta
    if not _check_step(alpha_init):
        alpha_init = 1.  # set to maximum step-size

    old_fval = fun(zt, *args)
    alpha_opt, q = _line_search_armijo(fun, zt, dt, gft, old_fval, args, c1, delta, alpha0)
    if alpha_opt is None:
        msg = "SLS failed to find a suitable learning rate, alpha outside range." \
              f"\ng*d={gft.dot(dt):.6f}, phi(0)={old_fval:.6f}"
        raise _LineSearchError(msg)

    return alpha_opt, q


def _check_step(alpha):
    # alpha_min = 0.
    alpha_min = 1e-14
    alpha_max = 1.

    return alpha_min < alpha <= alpha_max


# %% MSL

def _momentum_correction(beta0, gft, dt, delta):
    """
    Momentum correction procedure, dataset not required
    Damp the momentum term until the next direction is descent
    Dataset not required

    Parameters
    ----------
    beta0 : int
        initial momentum term.
    gft : numpy.ndarray
        samples gradient w.r.t. z_t.
    dt : numpy.ndarray
        previous iteration direction
        if t==0 is the null direction
    delta : float
        damping factor.

    Returns
    -------
    d : numpy.ndarray
        current iteration direction
    """

    max_iter = 100  # momentum correction maximum iterations
    beta1 = beta0    # starting momentum term

    def phi(beta):
        return -((1. - beta) * gft + beta * dt)

    d1 = phi(beta1)  # starting direction

    q = 0  # momentum term rejections counter
    while _check_momentum(beta1):

        q += 1

        if q >= max_iter:
            raise _MomentumCorrectionError("Maximum number of momentum corrections " +
                                           f"iterations has been exceeded. q={q}")

        # TODO: check something like if the correction is improving
        # like if it's decreasing

        # give more importance to the negative gradient
        beta1 *= delta   # reduce momentum term
        d1 = phi(beta1)  # update direction

        if _check_descent(gft, d1):
            return d1, q

        # q += 1

    # Failed to find a suitable momentum term
    return None, q


def _momentum_restart(beta0, gft, dt):
    """
    Momentum restart procedure, dataset not required
    Check if the next direction is descent
    Dataset not required

    Parameters
    ----------
    beta0 : int
        initial momentum term
    gft: numpy.ndarray
        w.r.t. z_t
    dt: numpy.ndarray
        previous iteration direction

    Returns
    -------
    dt : numpy.ndarray
        current iteration direction
    """

    # direction to check
    d1 = -((1. - beta0) * gft + beta0 * dt)

    # perhaps a check on the restarted direction is needed
    d2, q = (d1, 0) if _check_descent(gft, d1) else (-(1. - beta0) * gft, 1)

    return d2, q


def _stochastic_momentum(solver, k, t, beta0, gft, dt, delta):
    """
    Parameters
    ----------
    solver : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    beta0 : TYPE
        DESCRIPTION.
    gft : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.
    delta : TYPE
        DESCRIPTION.

    Raises
    ------
    _MomentumCorrectionError

    Returns
    -------
    d : array_like
    """

    if solver == "MSL-SGDM-C":
        dt, q = _momentum_correction(beta0, gft, dt, delta)
        if dt is None:
            msg = "MSL failed to find a suitable momentum term, beta outside range." \
                  f"\ng*d={gft.dot(dt)}"
            raise _MomentumCorrectionError(msg)

    elif solver == "MSL-SGDM-R":
        dt, q = _momentum_restart(beta0, gft, dt)

    return dt, q


def _check_momentum(beta):
    # beta_min = 0.
    beta_min = 1e-14
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


# %% Adam

def _adams(solver, mt, vt, gft, t):
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
    I = np.eye(vt.size)

    # vector m^t and v^t
    mt = beta1 * mt + (1 - beta1) * gft
    vt = beta2 * vt + (1 - beta2) * np.square(gft)

    # bias correction
    mtcap = mt / (1 - beta1**(t + 1))
    vtcap = vt / (1 - beta2**(t + 1))

    # TODO: Adamax routine

    # matrix inversion and direction computation
    Vt = np.linalg.inv(np.diag(np.sqrt(vtcap)) + eps * I)
    dt = -np.dot(Vt, mtcap)

    # MSL-Adam restart routine
    if solver == "MSL-Adam":
        if not _check_descent(gft, dt):
            # initial moments
            m0 = np.zeros_like(mt)
            v0 = np.zeros_like(vt)
            # recompute moments
            mt = beta1 * m0 + (1. - beta1) * gft
            vt = beta2 * v0 + (1. - beta2) * np.square(gft)
            # bias correction
            mtcap = mt / (1. - beta1**(t + 1))
            vtcap = vt / (1. - beta2**(t + 1))
            # inverse matrix and recompute direction
            Vt = np.linalg.inv(np.diag(np.sqrt(vtcap)) + eps * I)
            dt = -np.dot(Vt, mtcap)

    return mt, vt, dt


# %% exception utils

class _SearchDirectionError(RuntimeError):
    pass

class _LineSearchError(RuntimeError):
    pass

class _MomentumTermError(RuntimeError):
    pass

class _MomentumCorrectionError(RuntimeError):
    pass
