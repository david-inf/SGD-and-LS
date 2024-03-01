# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import accuracy_score

from solvers import (l_bfgs_b, newton_cg, cg, sgd_m, sgd_sls)
from solvers_utils import sigmoid


class LogisticRegression():
    def __init__(self, solver="L-BFGS", C=1):
        self.C = C  # lambda
        self.solver = solver

        self.opt_result = None

        self.coef_ = None
        self.loss = None
        self.loss_seq = None
        self.grad = None

        self.accuracy_train = None
        self.accuracy_test = None

    def fit(self, dataset=(), max_epochs=200, batch_size=16, step_size=1, momentum=0, stop=0):
        # dataset = (X_train_X, y_train_X, X_test_X, y_test_X)
        X_train = dataset[0]
        y_train = dataset[1]
        X_test = dataset[2]
        y_test = dataset[3]

        w0 = (1 + 1) * np.random.default_rng(42).random(X_train.shape[1]) - 1

        # dictionary of callables
        solver_dict = {
            "L-BFGS": l_bfgs_b, "Newton-CG": newton_cg, "CG": cg,
            "SGD-Fixed": sgd_m, "SGD-Decreasing": sgd_m, "SGDM": sgd_m,
            "SGD-Armijo": sgd_sls, "MSL-SGDM-C": sgd_sls, "MSL-SGDM-R": sgd_sls}

        if self.solver in ("L-BFGS", "Newton-CG", "CG"):
            model = solver_dict[self.solver](w0, X_train, y_train, self.C)

            self.opt_result = model
            self.coef_ = model.x
            self.loss = model.fun
            self.grad = np.linalg.norm(model.jac)

        elif self.solver in ("SGD-Fixed", "SGD-Decreasing", "SGDM"):
            model = solver_dict[self.solver](w0, X_train, y_train, self.C,
                batch_size, step_size, momentum, max_epochs, self.solver, stop)
            # sgd_m(w0, X, y, lam, M, alpha0, beta0, epochs, solver)

            self.opt_result = model
            self.coef_ = model.x
            self.loss = model.fun
            self.grad = np.linalg.norm(model.jac)
            self.loss_seq = model.fun_per_it# / N

        elif self.solver in ("SGD-Armijo", "MSL-SGDM-C", "MSL-SGDM-R"):
            model = solver_dict[self.solver](w0, X_train, y_train, self.C,
                batch_size, step_size, momentum, max_epochs, self.solver, stop)
            # sgd_sls(w0, X, y, lam, M, alpha0, beta0, epochs, solver)

            self.opt_result = model
            self.coef_ = model.x
            self.loss = model.fun
            self.grad = np.linalg.norm(model.jac)
            self.loss_seq = model.fun_per_it# / N

        self.accuracy_train = accuracy_score(y_train, self.predict(X_train))
        self.accuracy_test = accuracy_score(y_test, self.predict(X_test))

        return self

    def predict(self, X, thresh=0.5):
        y_proba = sigmoid(np.dot(X, self.coef_))

        y_pred = y_proba.copy()
        y_pred[y_pred > thresh] = 1
        y_pred[y_pred <= thresh] = -1

        return y_pred


# class LinearRegression():
#     def __init__(self, solver="L-BFGS", C=1):
#         self.C = C  # lambda
#         self.solver = solver

#         self.opt_result = None

#         self.coef_ = None
#         self.loss = None
#         self.loss_seq = None
#         self.grad = None
