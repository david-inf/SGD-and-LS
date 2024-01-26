# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 16:10:31 2024

@author: Utente
"""

import numpy as np
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
    def __init__(self, tol=1e-3, bias=False, solver="miniGD-fixed",
                 epochs=100, minibatch_size=5, regul_coef=0.5):
        self.tol = tol
        self.bias = bias
        self.solver = solver
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.regul_coef = regul_coef
        self.coef_ = None
        self.obj = None
        self.grad = None

    def fit(self, X, y, w0):
        if self.solver == "miniGD-fixed":
            self.coef_ = miniGD_fixed(logistic,
                            logistic_der, X, y, self.minibatch_size,
                            self.regul_coef, w0, epochs=self.epochs)
            return self
        elif self.solver == "miniGD-decreasing":
            self.coef_ , self.obj, self.grad = miniGD_decreasing(logistic,
                            logistic_der, X, y, self.minibatch_size,
                            self.regul_coef, w0, epochs=self.epochs)
            return self
