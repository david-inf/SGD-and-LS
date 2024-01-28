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
    def __init__(self, minibatch_size=1, tol=1e-3, bias=True,
                 solver="miniGD-fixed", epochs=300, regul_coef=0.5):
        self.minibatch_size = minibatch_size
        self.tol = tol
        self.bias = bias
        self.solver = solver
        self.epochs = epochs
        self.regul_coef = regul_coef
        self.bias = bias
        self.coef_ = None
        self.coef_seq = None
        self.obj = None
        self.obj_seq = None
        self.grad = None
        self.grad_seq = None
        self.benchmark = None

    def logistic(self, w, X, y):
        N = X.shape[0]
        loss = 0  # sum of loss over singles dataset examples
        for i in range(N):
            loss += np.log(1 + np.exp(- y[i] * np.dot(np.transpose(w), X[i,:])))
        regul = self.regul_coef * np.linalg.norm(w) ** 2
        return loss + regul

    def logistic_der(self, w, X, y):
        N = X.shape[0]
        r = np.zeros(N)
        for i in range(N):
            r[i] = - y[i] * sigmoid(- y[i] * np.dot(np.transpose(w), X[i,:]))
        loss_der = np.dot(np.transpose(X), r)
        # r = - y * sigmoid(-y * np.dot(X, w))
        # if X.ndim == 2:  # full gradient
        #     loss_der = np.matmul(np.transpose(X), r)
        # else:  # single example gradient
        #     loss_der = np.dot(X, r)
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

    def fit(self, X, y, w0, alpha=0.7):
        N = X.shape[0]  # dataset examples
        X = np.hstack((np.ones((N,1)),X.values))  # add constant column
        if self.solver == "scipy":
            # self.benchmark = optimalSolver(self.logistic, self.logistic_der, X, y, w0)
            return self
        elif self.solver == "miniGD-fixed":
            self.coef_seq, self.obj_seq, self.grad_seq = miniGD_fixed(
                self.logistic, self.logistic_der, X, y, self.minibatch_size, w0,
                self.regul_coef, alpha, self.tol, self.epochs, self.bias)
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
