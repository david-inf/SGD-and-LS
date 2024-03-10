# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.optimize import minimize

from solvers import sgd
from solvers_utils import sigmoid, logistic, logistic_der, f_and_df_log, logistic_hess
from solvers_utils import linear, linear_der, f_and_df_linear, linear_hess


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
        # dataset = (X_train, y_train, X_test, y_test)
        X_train = dataset[0]
        y_train = dataset[1]
        X_test = dataset[2]
        y_test = dataset[3]

        w0 = (1 + 1) * np.random.default_rng(42).random(X_train.shape[1]) - 1

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

            model = sgd(w0, X_train, y_train, self.C, batch_size, step_size,
                        momentum, max_epochs, self.solver, stop,
                        logistic, logistic_der, f_and_df_log)

            self.opt_result = model
            self.coef_ = model.x
            self.fun = model.fun
            self.grad = np.linalg.norm(model.jac)
            self.fun_seq = model.fun_per_epoch

        elif self.solver in ("SGD-Armijo", "MSL-SGDM-C", "MSL-SGDM-R"):
            # model = solver_dict[self.solver](w0, X_train, y_train, self.C,
            #     batch_size, step_size, momentum, max_epochs, self.solver, stop)
            # sgd_sls(w0, X, y, lam, M, alpha0, beta0, epochs, solver)

            model = sgd(w0, X_train, y_train, self.C, batch_size, step_size,
                        momentum, max_epochs, self.solver, stop,
                        logistic, logistic_der, f_and_df_log)

            self.opt_result = model
            self.coef_ = model.x
            self.fun = model.fun
            self.grad = np.linalg.norm(model.jac)
            self.fun_seq = model.fun_per_epoch

        self.accuracy_train = accuracy_score(y_train, self.predict(X_train))
        self.accuracy_test = accuracy_score(y_test, self.predict(X_test))

        return self


    def predict(self, X, thresh=0.5):
        y_proba = sigmoid(np.dot(X, self.coef_))

        y_pred = y_proba.copy()
        y_pred[y_pred > thresh] = 1
        y_pred[y_pred <= thresh] = -1

        return y_pred


    def __str__(self):
        return (f"Train score: {self.accuracy_train}" +
                f"\nTest score: {self.accuracy_test}" +
                f"\nObjective function: {self.fun:.6f}" +
                f"\nGrad norm: {self.grad:.6f}" +
                f"\nSol norm: {np.linalg.norm(self.coef_):.6f}")


class LinearRegression():
    def __init__(self, solver="L-BFGS-B", C=1):
        self.C = C  # lambda
        self.solver = solver

        self.opt_result = None

        self.coef_ = None
        self.fun = None
        self.fun_seq = None
        self.grad = None

        self.mse = None

    def fit(self, dataset=(), max_epochs=200, batch_size=16, step_size=1, momentum=0, stop=0):
        # dataset = (X_train, y_train, X_test, y_test)
        X_train = dataset[0]
        y_train = dataset[1]
        X_test = dataset[2]
        y_test = dataset[3]

        w0 = (1 + 1) * np.random.default_rng(42).random(X_train.shape[1]) - 1

        if self.solver in ("L-BFGS-B", "CG"):
            model = minimize(linear, w0, args=(X_train, y_train, self.C),
                             method=self.solver, jac=linear_der, bounds=None)

            self.opt_result = model
            self.coef_ = model.x
            self.fun = model.fun
            self.grad = np.linalg.norm(model.jac)

        elif self.solver == "Newton-CG":
            model = minimize(linear, w0, args=(X_train, y_train, self.C),
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

            model = sgd(w0, X_train, y_train, self.C, batch_size, step_size,
                        momentum, max_epochs, self.solver, stop,
                        linear, linear_der, f_and_df_linear)

            self.opt_result = model
            self.coef_ = model.x
            self.fun = model.fun
            self.grad = np.linalg.norm(model.jac)
            self.fun_seq = model.fun_per_epoch

        elif self.solver in ("SGD-Armijo", "MSL-SGDM-C", "MSL-SGDM-R"):
            # model = solver_dict[self.solver](w0, X_train, y_train, self.C,
            #     batch_size, step_size, momentum, max_epochs, self.solver, stop)
            # sgd_sls(w0, X, y, lam, M, alpha0, beta0, epochs, solver)

            model = sgd(w0, X_train, y_train, self.C, batch_size, step_size,
                        momentum, max_epochs, self.solver, stop,
                        linear, linear_der, f_and_df_linear)

            self.opt_result = model
            self.coef_ = model.x
            self.fun = model.fun
            self.grad = np.linalg.norm(model.jac)
            self.fun_seq = model.fun_per_epoch

        self.mse = mean_squared_error(y_test, self.predict(X_test))

        return self


    def predict(self, X):
        y_pred = np.dot(X, self.coef_)

        return y_pred


    def __str__(self):
        return (f"MSE: {self.mse:.6f}" +
                f"\nObjective function: {self.fun:.6f}" +
                f"\nGrad norm: {self.grad:.6f}" +
                f"\nSol norm: {np.linalg.norm(self.coef_):.6f}")
