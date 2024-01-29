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
        self.solver_message = None

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
        if X.ndim == 1:
            r_i = - y * sigmoid(- y * np.dot(w.T, X))
            loss_der = np.dot(X.T, r_i)
        elif X.ndim == 2:
            N = X.shape[0]
            r = np.zeros(N)
            for i in range(N):
                r[i] = - y[i] * sigmoid(- y[i] * np.dot(w.T, X[i,:]))
            loss_der = np.dot(np.transpose(X), r)
        regul_der = self.regul_coef * 2 * w
        return loss_der + regul_der

    # def logistic_hess(self, X, y, w, lam=0.5):
    #     D = np.zeros((X.shape[0], X.shape[0]))
    #     for i in range(X.shape[0]):
    #         D[i, i] = sigmoid(y[i] * np.dot(w, X[i, :])) * \
    #             sigmoid(-y[i] * np.dot(w, X[i, :]))
    #     loss_hess = np.matmul(np.matmul(np.transpose(X), D), X)
    #     regul_hess = 2 * lam * np.eye(X.shape[1])
    #     return loss_hess + regul_hess

    def fit(self, X, y, w0, learning_rate=0.09):
        if self.bias:
            N = X.shape[0]  # dataset examples
            X = np.hstack((np.ones((N,1)),X))  # add constant column
            if X.shape[1] != w0.shape:  # initial guess for bias not given
                w0 = np.insert(w0, 0, 0)  # add initial guess for bias
        elif self.solver == "scipy":
            self.benchmark_solver = optimalSolver(
                self.logistic, self.logistic_der, w0, X, y)
            return self
        elif self.solver == "miniGD-fixed":
            self.coef_seq, self.obj_seq, self.grad_seq, self.solver_message = miniGD_fixed(
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
