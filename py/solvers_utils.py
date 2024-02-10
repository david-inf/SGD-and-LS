
import numpy as np


def sigmoid(z):
    """
    Parameters
    ----------
    z : vector or scalar

    Returns
    -------
    vector or scalar like z.shape

    """
    return 1 / (1 + np.exp(-z))


# def call_f(w0, X, y, args=()):
#     return logistic(w0, X, y, *args)


def logistic(w, X, y, lam=1):
    """
    Parameters
    ----------
    w : vector
    X : matrix or vector
    y : vector or scalar
    lam : scalar

    Returns
    -------
    scalar
    """
    # N = X.shape[0]
    L_vec = np.log(1 + np.exp(- y * np.dot(X, w)))
    L = np.sum(L_vec)  # loss
    O = 0.5 * np.linalg.norm(w) ** 2  # regularizer
    return L + lam * O


def logistic_der(w, X, y, lam=1, coeff=1):
    """
    Parameters
    ----------
    w : vector
    X : matrix or vector
    y : vector or scalar
    lam : scalar

    Returns
    -------
    vector like w.shape[0]
    """
    # N = X.shape[0]
    z = - y * np.dot(X, w)
    r = - y * sigmoid(z)
    dL = np.dot(r, X)
    dO = w
    return dL + coeff * lam * dO


def f_and_df(w, X, y, lam=1):
    # fun, jac = f_and_df()
    # fun, _ = f_and_df()
    # _, jac = f_and_df()
    # N = X.shape[0]
    z = - y * np.dot(X, w)  # once for twice
    return (np.sum(np.log(1 + np.exp(z))) + lam * 0.5 * np.linalg.norm(w) ** 2,  # objective
            np.dot(- y * sigmoid(z), X) + lam * w)  # gradient


# def f_and_dfnorm(w, X, y, lam=1):
#     # N = X.shape[0]
#     z = - y * np.dot(X, w)  # compute one time instead on two
#     return (np.sum(np.log(1 + np.exp(z))) + lam * np.linalg.norm(w) ** 2,  # objective
#             np.linalg.norm(np.dot(- y * sigmoid(z), X) + lam * w))  # gradient norm


def logistic_hess(w, X, y, lam=1):
    """
    Parameters
    ----------
    w : vector
    X : matrix or vector
    y : vector or scalar
    coeff : scalar

    Returns
    -------
    Matrix like (w.shape[0], w.shape[0])
    """
    z = y * np.dot(X, w)  # vector or scalar
    if X.ndim == 2:
        D = np.diag(sigmoid(z) * sigmoid(- z))
    else:
        D = sigmoid(z) * sigmoid(- z)
    # if z scalar -> D scalar
    # if z vector -> D diagnonal matrix
    # D = sigmoid(z) * sigmoid(- z)
    ddL = np.dot(np.dot(D, X).T, X)
    ddO = lam * np.eye(w.shape[0])
    return ddL + ddO
