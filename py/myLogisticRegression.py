# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 16:10:31 2024

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
from MinibatchGD1 import miniGD_fixed, miniGD_decreasing, miniGD_armijo, optimalSolver

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class myLogRegr():
    def __init__(self, minibatch_size=1, tol=1e-4, bias=True,
                 solver="miniGD-fixed", epochs=100, regul_coef=1):
        self.minibatch_size = minibatch_size
        self.tol = tol
        self.bias = bias
        self.solver = solver
        self.epochs = epochs
        self.regul_coef = regul_coef
        self.bias = bias
        self.coef_ = None
        self.coef_seq = None
        self.obj_ = None
        self.obj_seq = None
        self.grad_ = None
        self.grad_seq = None
        self.benchmark_solver = None
        self.solver_message = ""
        self.opt_epochs = 0

    def logistic(self, w, X, y):
        # w : vector
        # X : matrix or vector
        # y : vector or scalar
        if X.ndim == 1:  # vector, a single example of the dataset
            # y should be a scalar
            loss = np.log(1 + np.exp(- y * np.dot(w.T, X)))
        elif X.ndim == 2: # matrix, N dataset examples
            N = X.shape[0]  # number of examples
            loss = 0  # sum of loss over singles dataset examples
            for i in range(N):
                loss += np.log(1 + np.exp(- y[i] * np.dot(w.T, X[i,:])))
        regul = self.regul_coef * np.linalg.norm(w) ** 2
        return loss + regul

    def logistic_der(self, w, X, y):
        # w : vector
        # X : matrix or vector
        # y : vector or scalar
        if X.ndim == 1:  # evaluate on a single example
            # y should be a scalar
            r_i = - y * sigmoid(- y * np.dot(w.T, X))
            loss_der = np.dot(X.T, r_i)
        elif X.ndim == 2:  # evaluate on the entire dataset
            N = X.shape[0]
            r = np.zeros(N)
            for i in range(N):
                r[i] = - y[i] * sigmoid(- y[i] * np.dot(w.T, X[i,:]))
            loss_der = np.dot(np.transpose(X), r)
        regul_der = self.regul_coef * 2 * w
        return loss_der + regul_der

    def logistic_hess(self, w, X, y):
        # w : vector
        # X : matrix or vector
        # y : vector or scalar
        if X.ndim == 1:  # evaluate on a single example
            # y should be a scalar
            sigma1 = sigmoid(y * np.dot(w.T, X))
            sigma2 = sigmoid(- y * np.dot(w.T, X))
            d_ii = sigma1 * sigma2
            loss_hess = d_ii * np.outer(X.T, X)
        elif X.ndim == 2:  # evaluate on the entire dataset
            D = np.zeros((X.shape[0], X.shape[0]))
            for i in range(X.shape[0]):
                sigma1 = sigmoid(y[i] * np.dot(w.T, X[i,:]))
                sigma2 = sigmoid(- y[i] * np.dot(w.T, X[i,:]))
                D[i,i] = sigma1 * sigma2
            loss_hess = np.matmul(np.matmul(X.T, D), X)
        regul_hess = 2 * self.regul_coef * np.eye(w.shape[0])
        return loss_hess + regul_hess

    def fit(self, X, y, w0, learning_rate=0.09):
        if self.bias:
            w0, X = self.checkBias(w0, X)
        if self.solver == "scipy":
            res = optimalSolver(self.logistic, self.logistic_der, w0, X, y)
            self.benchmark_solver = res
            self.coef_ = res.x
            self.solver_message = res.message
            self.obj_ = res.fun
            self.grad_ = res.jac
            return self
        elif self.solver == "miniGD-fixed":
            self.coef_seq, self.obj_seq, self.grad_seq = miniGD_fixed(
                self.logistic, self.logistic_der, X, y, self.minibatch_size,
                w0, self.regul_coef, self.tol, self.epochs, learning_rate)
            self.coef_ = self.coef_seq[-1]
            self.obj_ = self.obj_seq[-1]
            self.grad_ = self.grad_seq[-1]
            return self
        # elif self.solver == "miniGD-decreasing":
        #     self.coef_seq, self.obj_seq, self.grad_seq = miniGD_decreasing(
        #         logistic, logistic_der, X, y, self.minibatch_size,
        #         self.regul_coef, w0, alpha0=alpha, epochs=self.epochs)
        #     return self
        # elif self.solver == "miniGD-armijo":
        #     self.coef_seq, self.obj_seq, self.grad_seq = miniGD_armijo(
        #         logistic, logistic_der, X, y, self.minibatch_size, w0)
        #     return self

    def predict(self, X, thresh=0.5):
        N = X.shape[0]  # dataset examples
        if self.bias:
            X = np.hstack((np.ones((N,1)), X))  # add constant column
        y_proba = sigmoid(np.dot(self.coef_, X.T))
        y_proba[y_proba > thresh] = 1
        y_proba[y_proba <= thresh] = -1
        return y_proba

    def plotDiagnostic(self):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(7, 2.5),
                                       layout="constrained")
        ax1.plot(self.obj_seq)
        ax1.set_title("Training loss against epochs")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Training loss")
        ax1.set_yscale("log")
        ax2.plot(self.grad_seq)
        ax2.set_title("Gradient norm against epochs")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Gradient norm")
        # ax2.set_ylim([0, 100])  # function goes out of range
        plt.show()
        return self

    def checkBias(self, w, X):
        N = X.shape[0]  # dataset examples
        X = np.hstack((np.ones((N,1)), X))  # add constant column
        if X.shape[1] != w.shape[0]:  # check for bias initial guess
            w = np.insert(w, 0, 0)  # add initial guess for bias
        return w, X









