# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import accuracy_score
from solvers import (l_bfgs_b, newton_cg, cg, sgd_m, sgd_sls)
from ml_utils import predict


class LogisticRegression():
    def __init__(self, C=1, solver="L-BFGS", epochs=200, tol=1e-4, minibatch=16):
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
        # self.runtime = None
        self.accuracy_train = None
        self.accuracy_test = None

    def fit(self, w0, X_train, y_train, X_test, y_test, step_size=1, momentum=0):
        N = X_train.shape[0]
        # dictionary of callables
        solver_dict = {"L-BFGS": l_bfgs_b, "Newton-CG": newton_cg, "CG": cg,
                       "SGD-Fixed": sgd_m, "SGD-Decreasing": sgd_m, "SGDM": sgd_m,
                        "SGD-Armijo": sgd_sls, "MSL-SGDM-C": sgd_sls,
                        "MSL-SGDM-R": sgd_sls}
        if self.solver in ["L-BFGS", "Newton-CG", "CG"]:
            model = solver_dict[self.solver](w0, X_train, y_train, self.C)
            self.opt_result = model
            self.coef_ = model.x
            self.loss = model.fun
            self.grad = np.linalg.norm(model.jac)
            self.epochs = None
            self.tol = None
            self.minibatch = None
        elif self.solver in ("SGD-Fixed", "SGD-Decreasing", "SGDM"):
            model = solver_dict[self.solver](w0, X_train, y_train, self.C,
                    self.minibatch, step_size, momentum,
                    self.epochs, self.tol, self.solver)
            self.opt_result = model
            self.coef_ = model.x
            self.loss = model.fun
            self.grad = np.linalg.norm(model.jac)
            self.loss_seq = model.fun_per_it / N
        elif self.solver in ("SGD-Armijo", "MSL-SGDM-C", "MSL-SGDM-R"):
            model = solver_dict[self.solver](w0, X_train, y_train, self.C,
                    self.minibatch, step_size, momentum,
                    self.epochs, self.tol, self.solver)
            self.opt_result = model
            self.coef_ = model.x
            self.loss = model.fun
            self.grad = np.linalg.norm(model.jac)
            self.loss_seq = model.fun_per_it / N
        self.accuracy_train = accuracy_score(y_train, self.predict(X_train))
        self.accuracy_test = accuracy_score(y_test, self.predict(X_test))
        return self

    def predict(self, X, thresh=0.5):
        return predict(X, self.coef_, thresh)
