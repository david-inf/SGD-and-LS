# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import accuracy_score#, r2_score
from scipy.optimize import minimize

from solvers import sgd_m, sgd_sls
from solvers_utils import (sigmoid, logistic, logistic_der, f_and_df_log,
                           logistic_hess)
from solvers_utils import linear, linear_der, linear_hess


class LogisticRegression():
    def __init__(self, solver="L-BFGS-B", C=1):
        self.C = C  # lambda
        self.solver = solver

        self.opt_result = None

        self.coef_ = None
        self.fun = None
        self.fun_seq = None
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
        # solver_dict = {
        #     # "L-BFGS-B": l_bfgs_b, "Newton-CG": newton_cg, "CG": cg,
        #     "SGD-Fixed": sgd_m, "SGD-Decreasing": sgd_m, "SGDM": sgd_m,
        #     "SGD-Armijo": sgd_sls, "MSL-SGDM-C": sgd_sls, "MSL-SGDM-R": sgd_sls}

        if self.solver in ("L-BFGS-B", "CG"):
            model = minimize(f_and_df_log, w0, args=(X_train, y_train, self.C),
                             method=self.solver, jac=True, bounds=None)

            self.opt_result = model
            self.coef_ = model.x
            self.fun = model.fun
            self.grad = np.linalg.norm(model.jac)

        elif self.solver == "Newton-CG":
            model = minimize(f_and_df_log, w0, args=(X_train, y_train, self.C),
                             method=self.solver, jac=True, hess=logistic_hess,
                             bounds=None)

            self.opt_result = model
            self.coef_ = model.x
            self.fun = model.fun
            self.grad = np.linalg.norm(model.jac)

        elif self.solver in ("SGD-Fixed", "SGD-Decreasing", "SGDM"):
            # model = solver_dict[self.solver](w0, X_train, y_train, self.C,
            #     batch_size, step_size, momentum, max_epochs, self.solver, stop)
            # sgd_m(w0, X, y, lam, M, alpha0, beta0, epochs, solver)

            model = sgd_m(w0, X_train, y_train, self.C, batch_size, step_size,
                          momentum, max_epochs, self.solver, stop,
                          logistic, logistic_der, f_and_df_log)

            self.opt_result = model
            self.coef_ = model.x
            self.fun = model.fun
            self.grad = np.linalg.norm(model.jac)
            self.fun_seq = model.fun_per_epoch

        # elif self.solver in ("SGD-Armijo", "MSL-SGDM-C", "MSL-SGDM-R"):
        #     model = solver_dict[self.solver](w0, X_train, y_train, self.C,
        #         batch_size, step_size, momentum, max_epochs, self.solver, stop)
        #     # sgd_sls(w0, X, y, lam, M, alpha0, beta0, epochs, solver)

        #     self.opt_result = model
        #     self.coef_ = model.x
        #     self.fun = model.fun
        #     self.grad = np.linalg.norm(model.jac)
        #     self.loss_seq = model.loss_per_epoch

        self.accuracy_train = accuracy_score(y_train, self.predict(X_train))
        self.accuracy_test = accuracy_score(y_test, self.predict(X_test))

        return self

    def predict(self, X, thresh=0.5):
        y_proba = sigmoid(np.dot(X, self.coef_))

        y_pred = y_proba.copy()
        y_pred[y_pred > thresh] = 1
        y_pred[y_pred <= thresh] = -1

        return y_pred


class LinearRegression():
    def __init__(self, solver="L-BFGS-B", C=1):
        self.C = C  # lambda
        self.solver = solver

        self.opt_result = None

        self.coef_ = None
        self.loss = None
        self.loss_seq = None
        self.grad = None

        self.score = None

    def fit(self, dataset=(), max_epochs=200, batch_size=16, step_size=1, momentum=0, stop=0):
        # dataset = (X_train_X, y_train_X, X_test_X, y_test_X)
        X = dataset[0]
        y = dataset[1]

        w0 = (1 + 1) * np.random.default_rng(42).random(X.shape[1]) - 1

        if self.solver in ("L-BFGS-B", "CG"):
            model = minimize(linear, w0, args=(X, y, self.C),
                             method=self.solver, jac=linear_der, bounds=None)

            self.opt_result = model
            self.coef_ = model.x
            self.fun = model.fun
            self.grad = np.linalg.norm(model.jac)

        elif self.solver == "Newton-CG":
            model = minimize(linear, w0, args=(X, y, self.C),
                             method=self.solver, jac=linear_der, hess=linear_hess,
                             bounds=None)

            self.opt_result = model
            self.coef_ = model.x
            self.fun = model.fun
            self.grad = np.linalg.norm(model.jac)

        elif self.solver in ("SGD-Fixed", "SGD-Decreasing", "SGDM"):
            # model = solver_dict[self.solver](w0, X_train, y_train, self.C,
            #     batch_size, step_size, momentum, max_epochs, self.solver, stop)
            # sgd_m(w0, X, y, lam, M, alpha0, beta0, epochs, solver)

            model = sgd_m(w0, X, X, self.C, batch_size, step_size,
                          momentum, max_epochs, self.solver, stop,
                          linear, linear_der, f_and_df_linear)

            self.opt_result = model
            self.coef_ = model.x
            self.fun = model.fun
            self.grad = np.linalg.norm(model.jac)
            self.fun_seq = model.fun_per_epoch

        # self.score = r2_score(y_true, y_pred)

        return self
