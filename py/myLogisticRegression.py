# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 16:10:31 2024

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
from MinibatchGD1 import miniGD_fixed, miniGD_decreasing


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logistic(X, y, w, lam=0.5):
    # TODO: add bias
    loss = np.sum(np.log(1 + np.exp(- y * np.dot(X, w))))
    regul = lam * np.linalg.norm(w)
    return loss + regul


def logistic_der(X, y, w, lam=0.5):
    # TODO: add bias
    r = - y * sigmoid(-y * np.dot(X, w))
    if X.ndim == 2:  # full gradient
        loss_der = np.matmul(np.transpose(X), r)
    else:  # single example gradient
        loss_der = np.dot(X, r)
    regul_der = lam * 2 * w
    return loss_der + regul_der


class myLogRegr():
    def __init__(self, minibatch_size, tol=1e-3, bias=False,
                 solver="miniGD-fixed", epochs=100, regul_coef=0.5):
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

    def fit(self, X, y, w0, alpha):
        if self.solver == "miniGD-fixed":
            self.coef_seq, self.obj_seq, self.grad_seq = miniGD_fixed(
                logistic, logistic_der, X, y, self.minibatch_size,
                self.regul_coef, w0, alpha=alpha, epochs=self.epochs)
            return self
        elif self.solver == "miniGD-decreasing":
            self.coef_seq, self.obj_seq, self.grad_seq = miniGD_decreasing(
                logistic, logistic_der, X, y, self.minibatch_size,
                self.regul_coef, w0, alpha0=alpha, epochs=self.epochs)
            return self
        elif self.solver == "miniGD-armijo":
            return self

    def plotDiagnostic(self):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(7,2.5),
                            layout="constrained")
        ax1.plot(self.obj_seq)
        ax1.set_title("Training loss against epochs")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Training loss")
        grad_norm_seq = []
        for grad_k in self.grad_seq:
            grad_norm_seq.append(np.linalg.norm(grad_k))
        ax2.plot(grad_norm_seq)
        ax2.set_title("Gradient norm against epochs")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Gradient norm")
        # ax2.set_ylim([0, 100])  # function goes out of range
        plt.show()

