# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import accuracy_score
from solvers import (l_bfgs_b, sgd_fixed, sgd_decreasing, sgd_armijo,
                     sgdm, msl_sgdm_c, msl_sgdm_r)
from ml_utils import predict


class LogisticRegression():
    def __init__(self, C=1, solver="L-BFGS", epochs=100, tol=1e-3, minibatch=16):
        self.C = C  # lambda
        self.solver = solver
        self.epochs = epochs
        self.tol = tol
        self.minibatch = minibatch
        self.opt_result = None
        self.coef_ = None
        self.loss = None
        self.loss_seq = None
        self.grad = None
        self.runtime = None
        self.accuracy_train = None
        self.accuracy_test = None

    def fit(self, w0, X_train, y_train, X_test, y_test, step_size=1, momentum=0):
        # dictionary of callables
        solver_dict = {"SGD-Fixed": sgd_fixed, "SGD-Decreasing": sgd_decreasing,
                       "SGD-Armijo": sgd_armijo, "SGDM": sgdm,
                       "MSL-SGDM-C": msl_sgdm_c, "MSL-SGDM-R": msl_sgdm_r}
        if self.solver == "L-BFGS":
            model = l_bfgs_b(w0, X_train, y_train, self.C)
            self.opt_result = model
            self.coef_ = model.x
            self.loss = model.fun
            self.grad = np.linalg.norm(model.jac)
        elif self.solver in ["SGD-Fixed", "SGD-Decreasing", "SGD-Armijo"]:
            model = solver_dict[self.solver](w0, X_train, y_train, self.C,
                    self.minibatch, step_size, self.epochs, self.tol)
            self.opt_result = model
            self.coef_ = model.x
            self.loss = model.fun
            self.grad = model.grad
            self.loss_seq = model.fun_per_it
            self.runtime = model.runtime
            # sgd_fixed(w0, X, y, lam, M, alpha, epochs, tol)
        elif self.solver in ["SGDM", "MSL-SGDM-C", "MSL-SGDM-R"]:
            model = solver_dict[self.solver](w0, X_train, y_train, self.C,
                    self.minibatch, step_size, momentum, self.epochs, self.tol)
            self.opt_result = model
            self.coef_ = model.x
            self.loss = model.fun
            self.grad = model.grad
            self.loss_seq = model.fun_per_it
            self.runtime = model.runtime
            # sgdm(w0, X, y, lam, M, alpha, beta, epochs, tol)
        self.accuracy_train = accuracy_score(y_train, self.predict(X_train))
        self.accuracy_test = accuracy_score(y_test, self.predict(X_test))
        return self

    def predict(self, X, thresh=0.5):
        return predict(X, self.coef_, thresh)
