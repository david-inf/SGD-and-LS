# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 22:36:20 2024

@author: Utente
"""

import numpy as np
# from numba import jit#, cuda
from scipy.optimize import minimize


def optimalSolver(fun, grad, w0, X, y):
    res = minimize(fun, w0, args=(X, y), method="L-BFGS-B", jac=grad,
                   bounds=None, options={"gtol":1e-4})
    return res

class myOptResult():
    def __init__(self, fun, grad, w0, minibatch_size, tolerance, epochs):
        self.M = minibatch_size
        self.tol = tolerance
        self.epochs = epochs
        self.fun = fun
        self.grad = grad
        self.w0 = w0

    # @jit(nopython=True)
    def miniGD_fixed(self, X, y, alpha):
        # fun and grad are callable
        # TODO: return an object
        # TODO: generate docstring
        # TODO: time counter
        # number of examples and features
        N, p = X.shape
        w_seq = np.zeros((self.epochs + 1, p))  # weights sequence, w\in\R^p
        w_seq[0,:] = self.w0
        fun_seq = np.zeros(self.epochs + 1)  # loss values sequence
        fun_seq[0] = self.fun(self.w0, X, y)  # full loss
        grad_seq = np.zeros(self.epochs + 1)  # gradients norm sequence
        grad_seq[0] = np.linalg.norm(self.grad(self.w0, X, y))  # full gradient
        k = 0  # epochs counter
        while grad_seq[k] > self.tol * (1 + np.absolute(fun_seq[k])) and k < self.epochs:
            ## Shuffle dataset
            batch = np.arange(N)  # dataset indices, reset every epoch
            rng = np.random.default_rng(k)  # set different seed every epoch
            rng.shuffle(batch)  # shuffle indices
            minibatches = np.array_split(batch, N / self.M)  # create the minibatches
            ## Approximate gradient and update (internal) weights
            y_seq = np.zeros((len(minibatches) + 1, p))  # internal weights sequence
            y_seq[0,:] = w_seq[k]
            for t in range(len(minibatches)):  # for minibatch in minibatches
            # for t, minibatch in enumerate(minibatches):
                ## Evaluate gradient approximation
                mini_grad = np.zeros(p)  # store true gradient approximation
                for j in minibatches[t]:
                # for j in minibatch:  # for index in minibatch indeces
                    # evaluate gradient on a single example j at weight t
                    mini_grad += self.grad(y_seq[t], X[j,:], y[j])
                mini_grad = mini_grad / self.M
                ## Update (internal) weigths
                y_tnext = y_seq[t,:] - alpha * mini_grad  # model internal update
                y_seq[t+1,:] = y_tnext  # internal weights update
            ## Update sequence
            k += 1
            w_seq[k,:] = y_tnext
            fun_seq[k] = self.fun(y_tnext, X, y)
            grad_seq[k] = np.linalg.norm(self.grad(y_tnext, X, y))
        self.message = ""
        if grad_seq[-1] <= self.tol * (1 + np.absolute(fun_seq[-1])):
            self.message += "Gradient under tolerance"
        if k >= self.epochs:
            self.message += "Max epochs exceeded"
        # return w_seq, fun_seq, grad_seq#, message, k
        # return w_seq[k,:], fun_seq[k], grad_seq[k], message
        self.sol = w_seq[k,:]
        self.fun = fun_seq[k]
        self.grad = grad_seq[k]
        return self


# @jit(nopython=True)
# def miniGD_decreasing(fun, grad, X, y, M, w0, lam, tol, epochs, alpha0):
#     # fun and grad are callable
#     N = X.shape[0]  # number of examples
#     p = X.shape[1]  # number of features
#     w_seq = [w0]  # weigth sequence, w\in\R^p
#     fun_seq = [fun(X, y, w0, lam)]
#     grad_seq = [grad(X, y, w0, lam)]
#     alpha_seq = alpha0 / (1 + np.arange(epochs - 1))
#     k = 0
#     while grad_seq[-1] > tol * (1 + np.absolute(fun_seq[-1])) and k <= epochs:
#         batch = np.arange(N)  # dataset indices
#         rng = np.random.default_rng(k)
#         rng.shuffle(batch)  # shuffle dataset indices
#         minibatches = np.array_split(batch, N / M)  # create the minibatches
#         y_seq = [w_seq[k]]  # internal epoch weights
#         for t in range(len(minibatches)):  # for minibatch in minibatches
#             mini_grad = np.zeros(p)  # store true gradient approximation
#             for j in minibatches[t]:  # for index in minibatch indeces
#                 # evaluate gradient on a single example j at weight t
#                 # mini_grad += logistic_der(X[j,:], y[j], y_seq[t], lam)
#                 mini_grad += grad(X[j,:], y[j], y_seq[t], lam)
#             y_tnext = y_seq[t] - alpha_seq[k] * mini_grad / M  # model internal update
#             y_seq.append(y_tnext)  # internal weights update
#         w_seq.append(y_seq[-1])  # weights update
#         fun_seq.append(fun(X, y, y_tnext, lam))
#         grad_seq.append(grad(X, y, y_tnext, lam))
#         k += 1
#     if grad_seq[-1] <= tol * (1 + np.absolute(fun_seq[-1])):
#         message = "Gradient under tolerance"
#     if k > epochs:
#         message = "Max epochs exceeded"
#     return w_seq, fun_seq, grad_seq, message, k


# def resetStep(N, alpha, alpha0, M, a, t, opt):
#     """
#     Parameters
#     ----------
#     N : int
#         Dataset examples
#     alpha : float
#         Previous iteration (mini-batch) step-size
#     alpha0 : float
#         Maximum step-size
#     M : int
#         Mini-batch size
#     a : float
#         arbitrary constant greater than 1, tunable parameter
#     t : int
#         Iteration index
#     opt : int
#         Step-size resetting type

#     Returns
#     -------
#     alpha : float
#         Resetted step-size. If opt==2 it is greater than the previous one.
#     """
#     if t == 0:
#         return alpha0
#     elif opt == 0:
#         return  alpha
#     elif opt == 1:
#         return alpha0
#     elif opt == 2:
#         return alpha * a ** (M / N)

# @jit(nopython=True)
# def miniGD_armijo(fun, grad, X, y, M, w0, lam=0.5,
#                   alpha0=1, tol=0.001, epochs=300,
#                   gamma=0.5, delta=0.5, bias=True):
#     N = X.shape[0]  # number of examples
#     p = X.shape[1]  # number of features
#     if bias:
#         X = np.column_stack((np.ones(N), X))
#         p += 1
#     w_seq = [w0]  # weigth sequence, w\in\R^p
#     fun_seq = [fun(X, y, w0, lam)]  # loss sequence
#     grad_seq = [np.linalg.norm(grad(X, y, w0, lam))]
#     k = 0  # epochs counter
#     while grad_seq[-1] > tol * (1 + np.absolute(fun_seq[-1])) and k <= epochs:
#         ## Shuffle dataset and create minibatches
#         batch = np.arange(N)  # dataset indices
#         rng = np.random.default_rng(k)  # set variable seed
#         rng.shuffle(batch)  # shuffle dataset indices
#         minibatches = np.array_split(batch, N / M)  # create minibatches
#         y_seq = [w_seq[k]]  # internal epoch weights update sequence
#         alpha_seq = [alpha0]  # step-size per minibatch
#         for t in range(len(minibatches)):  # for minibatch in minibatches
#             ## Compute true gradient approximation (parallelizzabile)
#             mini_grad = np.zeros(p)
#             for j in minibatches[t]:  # for index in minibatch indeces
#                 # evaluate gradient on a single example j at weights t
#                 mini_grad += grad(X[j,:], y[j], y_seq[t], lam)
#             mini_grad = mini_grad / M  # true gradient approximation
#             ## Reset step-size
#             alpha = resetStep(N, alpha_seq[-1], alpha0, M, 100, t, 2)
#             ## Armijo
#             q = 0  # step-size rejections counter
#             y_tnext = y_seq[t] - alpha * mini_grad
#             while fun(X, y, y_tnext, lam) > fun(X, y, y_seq[t], lam) - gamma * alpha * np.linalg.norm(grad(X, y, y_seq[t])) ** 2:
#                 alpha = delta * alpha
#                 y_tnext = y_seq[t] - alpha * mini_grad
#                 q += 1  # q uodates of the step-size
#             ## Internal weights update
#             alpha_seq.append(alpha)  # accepted step-size
#             y_seq.append(y_tnext)
#         ## Weights update
#         w_seq.append(y_seq[-1])  # weights update
#         fun_seq.append(fun(X, y, w_seq[-1], lam))
#         grad_seq.append(np.linalg.norm(grad(X, y, w_seq[-1], lam)))
#         k += 1
#     # return f"Value: {w_seq[-1]}\nIterations: {k}"
#     # TODO: return epochs counter, iterations over minibatches
#     return w_seq, fun_seq, grad_seq
