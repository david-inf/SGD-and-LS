# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la

# standard status messages of optimizers
_status_message = {"success": "Optimization terminated successfully.",
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   "maxiter": "Maximum number of epochs has been exceeded.",
                   "nan": "NaN result encountered.",
                   'out_of_bounds': 'The result is outside of the provided '
                                    'bounds.'}

def _check_errors(warnflag, k, max_epochs, gfk, fk, wk):
    if warnflag == 1:
        msg = "Something went wrong in line search or momentum correction"
    elif k >= max_epochs:
        warnflag = 2
        msg = _status_message["maxiter"]
    elif np.isnan(gfk).any() or np.isnan(fk) or np.isnan(wk).any():
        warnflag = 3
        msg = _status_message["nan"]
    else:
        msg = _status_message["success"]

    return msg, warnflag


def _check_descent(gfk, dk):
    # 0. is the mathematical formulation
    # need to check also the magnitude for a significant update
    val = gfk.dot(dk)  # float
    return val < -1e-8


def _shuffle_dataset(N, M, generator):
    """
    Shuffle dataset indices, dataset not required

    Parameters
    ----------
    N : int
        dataset size.
    k : int
        epochs counter.
    M : int
        current mini-batch size.

    Returns
    -------
    minibatches : list of numpy.ndarray of numpy.int32
    """

    batch = np.arange(N)  # dataset indices

    generator.shuffle(batch)  # shuffle indices

    # array_split is expensive, consider another strategy
    minibatches = np.array_split(batch, N / M)  # create the minibatches

    return minibatches


def _stopping(fk, gfk, nit, maxiter, criterion):
    """Stopping criterion

    Parameters
    ----------
    fk : float
        Objective function value.
    gfk : array_like
        Function gradient value.
    nit : int
        Current number of epochs.
    maxiter : int
        Maximum number of epochs.
    criterion : int
        Rule to use.

    Returns
    -------
    stop : boolean
    """

    tol = 1e-3
    stop = False

    if criterion == 0:
        stop = nit < maxiter

    elif criterion == 1:
        stop = (la.norm(gfk) > tol) and (nit < maxiter)

    elif criterion == 2:
        # return (np.linalg.norm(grad_k, np.inf) > tol) and (nit < max_iter)
        stop = (la.norm(gfk) > tol * (1 + fk)) and (nit < maxiter)

    return stop
