# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 22:36:20 2024

@author: Utente
"""

"""
Mini-batch Gradient Descent:
1. fized step-size
2. decreasing step-size
3. armijo line search
"""

import numpy as np
# from numba import jit, cuda

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))


# def logistic(X, y, w, lam=0.5):
#     # TODO: add bias
#     loss = np.sum(np.log(1 + np.exp(- y * np.dot(X, w))))
#     regul = lam * np.linalg.norm(w)
#     return loss + regul


# def logistic_der(X, y, w, lam=0.5):
#     # TODO: add bias
#     r = - y * sigmoid(-y * np.dot(X, w))
#     if X.ndim == 2: # full gradient
#         loss_der = np.matmul(np.transpose(X), r)
#     else: # single example gradient
#         loss_der = np.dot(X, r)
#     regul_der = lam * 2 * w
#     return loss_der + regul_der

# def prova(fun, X, y, w):
#     return fun(X, y, w)


# @jit(target_backend="cuda", nopython=True)
def miniGD_fixed(fun, grad, X, y, M, lam, w0,
                 alpha=0.7, tol=0.001, epochs=100):
    # fun and grad are callable
    # TODO: generate docstring
    # TODO: routing for adding bias with constant column in X
    N = X.shape[0]  # number of examples
    p = X.shape[1]  # number of features
    # batch = np.arange(N)  # dataset indices
    w_seq = [w0]  # weigth sequence, w\in\R^p
    fun_seq = [fun(X, y, w0, lam)]
    grad_seq = [grad(X, y, w0, lam)]
    k = 0
    # while np.linalg.norm(logistic_der(X, y, w_seq[k], lam)) > tol and k <= epochs:
    # while np.linalg.norm(grad(X, y, w_seq[k], lam)) > tol and k <= epochs:
    while k < epochs - 1:  # for performance evaluations
        batch = np.arange(N)  # dataset indices, reset every epoch
        rng = np.random.default_rng(k)
        rng.shuffle(batch)  # shuffle dataset indices
        minibatches = np.array_split(batch, N / M)  # create the minibatches
        y_seq = [w_seq[k]]  # internal epoch weights
        for t in range(len(minibatches)):  # for minibatch in minibatches
            mini_grad = np.zeros(p)  # store true gradient approximation
            for j in minibatches[t]:  # for index in minibatch indeces
                # evaluate gradient on a single example j at weight t
                # mini_grad += logistic_der(X[j,:], y[j], y_seq[t], lam)
                mini_grad += grad(X[j,:], y[j], y_seq[t], lam)
            mini_grad = mini_grad / M
            y_tnext = y_seq[t] - alpha * mini_grad  # model internal update
            y_seq.append(y_tnext)  # internal weights update
        w_seq.append(y_tnext)  # weights update
        fun_seq.append(fun(X, y, y_tnext, lam))
        grad_seq.append(grad(X, y, y_tnext, lam))
        k += 1
    # return f"Value: {w_seq[-1]}\nIterations: {k}"
    return w_seq, fun_seq, grad_seq


# @jit(target="cuda", nopython=True)
def miniGD_decreasing(fun, grad, X, y, M, lam, w0,
                      alpha0=5, tol=0.001, epochs=100):
    # fun and grad are callable
    N = X.shape[0]  # number of examples
    p = X.shape[1]  # number of features
    w_seq = [w0]  # weigth sequence, w\in\R^p
    fun_seq = [fun(X, y, w0, lam)]
    grad_seq = [grad(X, y, w0, lam)]
    alpha_seq = alpha0 / (1 + np.arange(epochs - 1))
    k = 0
    # while np.linalg.norm(logistic_der(X, y, w_seq[k], lam)) > tol and k <= epochs:
    # while np.linalg.norm(grad(X, y, w_seq[k], lam)) > tol and k <= epochs:
    while k < epochs - 1:  # for performance evaluations
        batch = np.arange(N)  # dataset indices
        rng = np.random.default_rng(k)
        rng.shuffle(batch)  # shuffle dataset indices
        minibatches = np.array_split(batch, N / M)  # create the minibatches
        y_seq = [w_seq[k]]  # internal epoch weights
        for t in range(len(minibatches)):  # for minibatch in minibatches
            mini_grad = np.zeros(p)  # store true gradient approximation
            for j in minibatches[t]:  # for index in minibatch indeces
                # evaluate gradient on a single example j at weight t
                # mini_grad += logistic_der(X[j,:], y[j], y_seq[t], lam)
                mini_grad += grad(X[j,:], y[j], y_seq[t], lam)
            y_tnext = y_seq[t] - alpha_seq[k] * mini_grad / M  # model internal update
            y_seq.append(y_tnext)  # internal weights update
        w_seq.append(y_seq[-1])  # weights update
        fun_seq.append(fun(X, y, y_tnext, lam))
        grad_seq.append(grad(X, y, y_tnext, lam))
        k += 1
    # return f"Value: {w_seq[-1]}\nIterations: {k}"
    return w_seq, fun_seq, grad_seq


# def resetStep(alpha, alpha0)

def miniGD_armijo(fun, grad, X, y, M, lam, w0,
                  alpha0=5, tol=0.001, epochs=100,
                  gamma=0.5, delta=0.5):
    N = X.shape[0]  # number of examples
    p = X.shape[1]  # number of features
    w_seq = [w0]  # weigth sequence, w\in\R^p
    fun_seq = [fun(X, y, w0, lam)]
    grad_seq = [grad(X, y, w0, lam)]
    k = 0
    while k < epochs - 1:  # for performance evaluations
        batch = np.arange(N)  # dataset indices
        rng = np.random.default_rng(k)
        rng.shuffle(batch)  # shuffle dataset indices
        minibatches = np.array_split(batch, N / M)  # create the minibatches
        y_seq = [w_seq[k]]  # internal epoch weights
        # alpha_seq = [alpha0]
        for t in range(len(minibatches)):  # for minibatch in minibatches
            mini_grad = np.zeros(p)  # store true gradient approximation
            for j in minibatches[t]:  # for index in minibatch indeces
                # evaluate gradient on a single example j at weight t
                mini_grad += grad(X[j,:], y[j], y_seq[t], lam)
            mini_grad = mini_grad / M
            ## Armijo
            # TODO: reset step-size
            alpha = alpha0
            q = 0
            y_tnext = y_seq[t] - alpha * mini_grad
            while fun(X, y, y_tnext, lam) > fun(X, y, y_seq[t], lam) - gamma * alpha * np.linalg.norm(grad(X, y, y_seq[t])) ** 2:
                alpha = delta * alpha
                q += 1
            y_seq.append(y_tnext)  # internal weights update
        w_seq.append(y_seq[-1])  # weights update
        fun_seq.append(fun(X, y, y_tnext, lam))
        grad_seq.append(grad(X, y, y_tnext, lam))
        k += 1
    # return f"Value: {w_seq[-1]}\nIterations: {k}"
    return w_seq, fun_seq, grad_seq
