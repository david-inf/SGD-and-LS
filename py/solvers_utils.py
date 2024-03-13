# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
from scipy.sparse import csr_matrix, isspmatrix_csr

# %% Logistic Regression

def sigmoid(z):
    """
    Sigmoid function

    Parameters
    ----------
    z : numpy.ndarray

    Returns
    -------
    numpy.float64
    """

    return 1 / (1 + np.exp(-z))


def logistic(w, X, y, lam):
    """
    Log-loss with l2 regularization

    Parameters
    ----------
    w : numpy.ndarray
        size p
    X : scipy.sparse.csr_matrix
        size Nxp
    y : numpy.ndarray
        size N
    lam : int

    Returns
    -------
    numpy.float64
    """

    samples = y.size  # number of samples

    # loss function
    # z = -y * np.dot(X, w)
    # loss = np.sum(np.logaddexp(0, z)) / samples

    # handle CSR loss function
    z = -y * X.dot(w)
    loss = np.sum(np.logaddexp(0, z)) / samples

    # regularizer term
    regul = 0.5 * la.norm(w) ** 2

    return loss + lam * regul


def logistic_der(w, X, y, lam):
    """
    Log-loss with l2 regularization derivative

    Parameters
    ----------
    w : numpy.ndarray
        size p
    X : scipy.sparse.csr_matrix
        size Nxp
    y : numpy.ndarray
        size N
    lam : int

    Returns
    -------
    numpy.ndarray of shape w.size
    """
    samples = y.size  # number of samples

    # loss function derivative
    # z = - y * np.dot(X, w)
    # loss_der = np.dot(-y * sigmoid(z), X) / samples

    # handle CSR loss function derivative
    z = -y * X.dot(w)
    loss_der = X.T.dot(-y * sigmoid(z)) / samples

    # regularizer term derivative
    regul_der = w

    return loss_der + lam * regul_der


def f_and_df_log(w, X, y, lam):
    """
    Log-loss with l2 regularization and its derivative

    Parameters
    ----------
    w : numpy.ndarray
        size p
    X : scipy.sparse.csr_matrix
        size Nxp
    y : numpy.ndarray
        size N
    lam : int

    Returns
    -------
    numpy.float64,
    numpy.ndarray of shape w.size
    """

    samples = y.size  # number of samples

    z = -y * X.dot(w)  # once for twice

    # handle CSR loss function and regularizer term
    loss = np.sum(np.logaddexp(0, z)) / samples
    regul = 0.5 * np.linalg.norm(w) ** 2

    # handle CSR loss function and regularizer term derivatives
    loss_der = X.T.dot(-y * sigmoid(z)) / samples
    regul_der = w

    return (loss + lam * regul,          # objective function
            loss_der + lam * regul_der)  # jacobian


def logistic_hess(w, X, y, lam):
    """
    Log-loss with l2 regularization hessian

    Parameters
    ----------
    w : numpy.ndarray
        size p
    X : scipy.sparse.csr_matrix
        size Nxp
    y : numpy.ndarray
        size N
    lam : int

    Returns
    -------
    numpy.ndarray of shape (w.size, w.size)
    """

    samples = y.size  # number of samples

    z = y * X.dot(w)  # once for twice

    # diagnonal matrix NxN
    D = csr_matrix(np.diag(sigmoid(z) * sigmoid(-z)))

    # handle CSR loss function hessian
    # loss_hess = np.dot(np.dot(D, X).T, X) / samples
    loss_hess = X.T.dot(D).dot(X) / samples

    # regularizer term hessian
    regul_hess = np.eye(w.size)

    return loss_hess.toarray() + lam * regul_hess


# %% Multiple linear regression
# TODO: handle CSR

def linear(w, X, y, lam):
    """ Quadratic-loss with l2 regularization """

    samples = y.size  # number of samples

    # loss function
    # loss = 0.5 * np.linalg.norm(np.dot(X, w) - y)**2 / samples
    XXw = np.dot(np.matmul(X.T, X), w)  # vector
    yX = np.dot(y, X)                   # vector
    loss = 0.5 * (np.dot(w, XXw) - 2 * np.dot(yX, w) + np.linalg.norm(y)**2) / samples

    # regularizer term
    regul = 0.5 * np.linalg.norm(w) ** 2

    return loss + lam * regul


def linear_der(w, X, y, lam):
    """ Quadratic-loss with l2 regularization derivative """
    samples = y.size  # number of samples

    # loss function derivative
    XXw = np.dot(np.matmul(X.T, X), w)  # vector
    yX = np.dot(y, X)                   # vector
    loss_der = (XXw - yX) / samples

    # regularizer term derivative
    regul_der = w

    return loss_der + lam * regul_der


def f_and_df_linear(w, X, y, lam):
    """ Quadratic-loss with l2 regularization and its derivative """
    samples = y.size  # number of samples

    XXw = np.dot(np.matmul(X.T, X), w)  # once for twice
    yX = np.dot(y, X)                   # once for twice

    # loss function and regularizer term
    loss = 0.5 * (np.dot(w, XXw) - 2 * np.dot(yX, w) + np.linalg.norm(y)**2) / samples
    regul = 0.5 * np.linalg.norm(w) ** 2

    # loss function and regularizer term derivatives
    loss_der = (XXw - yX) / samples
    regul_der = w

    return (loss + lam * regul,          # objective function
            loss_der + lam * regul_der)  # jacobian


def linear_hess(w, X, y, lam):
    """ Quadratic-loss with l2 regularization hessian """
    samples = y.size  # number of samples

    # loss function hessian
    loss_hess = np.matmul(X.T, X) / samples

    # regularizer term hessian
    regul_hess = np.eye(w.size)

    return loss_hess + lam * regul_hess
