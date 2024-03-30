# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error
from scipy.optimize import minimize

from solvers import minibatch_gd
from solvers_utils import sigmoid, logistic, logistic_der, f_and_df_log, logistic_hess
from solvers_utils import linear, linear_der, f_and_df_linear, linear_hess


class LogisticRegression():
    def __init__(self, solver="L-BFGS-B", C=1):
        """
        Parameters
        ----------
        solver : string, optional
            solver to be used. The default is "L-BFGS-B".
        C : int, optional
            regularization term. The default is 1.

        Returns
        -------
        None.
        """

        self.C = C  # lambda
        self.solver = solver

        self.opt_result = None  # OptimizeResult

        self.coef_ = None
        self.fun = None
        self.fun_seq = None
        self.grad = None

        self.metrics_train = None  # list of floats
        self.metrics_test = None   # list of floats


    def fit(self, dataset=(), batch_size=16, step_size=0.1, momentum=0, stop=1,
            max_epochs=600, damp_armijo=0.5, gamma_armijo=0.001, damp_momentum=0.5,
            **kwargs):
        """
        Parameters
        ----------
        dataset : tuple, optional
            (X_train, y_train, X_test, y_test) The default is ().
        max_epochs : int, optional
            maximum number of epochs. The default is 200.
        batch_size : int, optional
            mini-batch size. The default is 16.
        step_size : float, optional
            initial learning rate. The default is 1.
        momentum : float, optional
            momentum term. The default is 0.
        stop : int, optional
            stopping criterion. The default is 0.

        Returns
        -------
        LogisticRegression
            fitted model.
        """

        sgd_variants = ("SGD-Fixed", "SGD-Decreasing", "SGDM",
                   "SGD-Armijo", "MSL-SGDM-C", "MSL-SGDM-R")

        X_train = dataset[0]  # scipy.sparse.csr_matrix
        y_train = dataset[1]  # numpy.ndarray
        X_test = dataset[2]   # scipy.sparse.csr_matrix
        y_test = dataset[3]   # numpy.ndarray

        w0 = (0.5 + 0.5) * np.random.default_rng(42).random(X_train.shape[1] - 1) - 0.5
        w0 = np.insert(w0, 0, 0)  # null bias

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

        elif self.solver in sgd_variants:
            model = minibatch_gd(w0, X_train, y_train, self.C, batch_size,
                                 step_size, momentum, max_epochs, self.solver,
                                 stop, damp_armijo, gamma_armijo, damp_momentum,
                                 logistic, logistic_der, f_and_df_log)

            self.opt_result = model
            self.coef_ = model.x
            self.fun = model.fun
            self.grad = np.linalg.norm(model.jac)
            self.fun_seq = model.fun_per_epoch

        self.metrics_train = [accuracy_score(y_train, self.predict(X_train)),
                              balanced_accuracy_score(y_train, self.predict(X_train))]
        self.metrics_test = [accuracy_score(y_test, self.predict(X_test)),
                             balanced_accuracy_score(y_test, self.predict(X_test))]

        return self


    def predict(self, X, thresh=0.5):
        """
        Logistic regression prediction
        Dataset required, handles CSR matrices

        Parameters
        ----------
        X : scipy.sparse.csr_matrix
            test dataset.
        thresh : float, optional
            classification threshold. The default is 0.5.

        Returns
        -------
        y_pred : numpy.ndarray
            array with predictions.
        """

        y_proba = sigmoid(X.dot(self.coef_))

        y_pred = np.where(y_proba > thresh, 1, -1)

        return y_pred


    def __str__(self):

        return (f"Solver: {self.solver}" +
                # f"\nTrain score: {self.metrics_train[0]}" +
                f"\nTest score: {self.metrics_test[0]}" +
                f"\nObjective function: {self.fun:.6f}" +
                f"\nGrad norm: {self.grad:.6f}" +
                f"\nSol norm: {np.linalg.norm(self.coef_):.6f}" +
                f"\nRun-time (seconds): {self.opt_result.runtime:.6f}" +
                f"\nEpochs: {self.opt_result.nit}")


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

    def fit(self, dataset=(), max_epochs=200, batch_size=16, step_size=1,
            momentum=0, stop=0, parallel=False):
        # dataset = (X_train, y_train, X_test, y_test)
        X_train = dataset[0]
        y_train = dataset[1]
        X_test = dataset[2]
        y_test = dataset[3]

        sgd_variants = ("SGD-Fixed", "SGD-Decreasing", "SGDM",
                   "SGD-Armijo", "MSL-SGDM-C", "MSL-SGDM-R")

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

        elif self.solver in sgd_variants:
            model = minibatch_gd(w0, X_train, y_train, self.C, batch_size,
                                 step_size, momentum, max_epochs, self.solver,
                                 stop, linear, linear_der, f_and_df_linear)

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
