# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

from solvers import minibatch_gd, batch_gd
from functions import sigmoid, logistic, logistic_der, f_and_df_log, logistic_hess
from functions import linear, linear_der, f_and_df_linear, linear_hess
from ml_utils import metrics_list


class LogisticRegression():
    def __init__(self, solver="L-BFGS-B", C=1):
        """
        Parameters
        ----------
        solver : string, optional
            Solver to be used. The default is "L-BFGS-B".
        C : int, optional
            Regularization coefficient. The default is 1.
        """

        self.C = C  # lambda
        self.solver = solver

        self.opt_result = None  # OptimizeResult

        self.coef_ = None  # model weights including intercept
        self.fun = None    # objective function value
        self.grad = None   # gradient norm value

        self.metrics_train = None  # list of floats
        self.metrics_test = None   # list of floats


    def fit(self, dataset, w0=None, batch_size=32, l_rate=0.1, momentum=0., stop=1,
            max_epochs=600, damp_armijo=0.5, c1_armijo=1e-5, damp_momentum=0.5,
            **kwargs):
        """
        Parameters
        ----------
        dataset : tuple, optional
            (X_train, y_train, X_test, y_test).
        batch_size : int, optional
            Mini-batch size. The default is 32.
        w0 : array_like
            Initial guess for algorithm.
        l_rate : float, optional
            Initial learning rate. The default is 0.1.
        momentum : float, optional
            Momentum term. The default is 0.
        stop : int, optional
            Stopping criterion. The default is 1, when reaches optimal solution.
            Other options are: 0 (run algorithm for specified epochs)
        max_epochs : int, optional
            Maximum number of epochs. The default is 600.
        damp_armijo : float, optional
            Damping for Armijo line search. The default is 0.5.
        gamma_armijo : float, optional
            Armijo condition aggressiveness. The default is 0.001.
        damp_momentum : TYPE, optional
            Damping for momentum correction procedure. The default is 0.5.

        Returns
        -------
        LogisticRegression
        """

        batch_variants = ("BatchGD-Fixed", "BatchGD-Decreasing", "BatchGDM",
                          "BatchGD-Armijo")

        sgd_variants = ("SGD-Fixed", "SGD-Decreasing", "SGDM",
                        "SGD-Armijo", "MSL-SGDM-C", "MSL-SGDM-R",
                        "Adam", "MSL-Adam")

        sgd_more_variants = ("Adam", "Adamax", "MSL-Adam")

        X_train = dataset[0]  # scipy.sparse.csr_matrix
        y_train = dataset[1]  # numpy.ndarray
        X_test = dataset[2]   # scipy.sparse.csr_matrix
        y_test = dataset[3]   # numpy.ndarray

        f_args = (X_train, y_train, self.C)  # objective function arguments

        # check on initial guess
        if w0 is None:
            w0 = (0.5 + 0.5) * np.random.default_rng(42).random(X_train.shape[1] - 1) - 0.5
            w0 = np.insert(w0, 0, 0.)  # null bias

        # select solver and run algorithm
        if self.solver in ("L-BFGS-B", "CG"):
            model = minimize(f_and_df_log, w0, args=f_args,
                             method=self.solver, jac=True, bounds=None)

        elif self.solver == "Newton-CG":
            model = minimize(f_and_df_log, w0, args=f_args,
                             method=self.solver, jac=True, hess=logistic_hess,
                             bounds=None)

        elif self.solver in batch_variants:
            model = batch_gd(logistic, w0, f_args, self.solver,
                             logistic_der, l_rate, momentum, max_epochs, stop)

        elif self.solver in sgd_variants:
            model = minibatch_gd(logistic, w0, f_args,
                                 self.solver, logistic_der, f_and_df_log,
                                 batch_size, l_rate, momentum, max_epochs,
                                 stop, damp_armijo, c1_armijo, damp_momentum)

        # check optimization result
        if not model.success:
            print(model.message)

            return self

        self.opt_result = model
        self.coef_ = model.x
        self.fun = model.fun
        self.grad = np.linalg.norm(model.jac)

        self.metrics_train = metrics_list(y_train, self.predict(X_train))
        self.metrics_test = metrics_list(y_test, self.predict(X_test))

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

        if self.opt_result is None:
            return "Optimization went wrong"

        return (f"Solver: {self.solver}" +
                f"\nTrain score: {self.metrics_train[0]:.6f}" +
                f"\nTest score: {self.metrics_test[0]:.6f}" +
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

    def fit(self, dataset=(), batch_size=16, step_size=0.1, momentum=0, stop=1,
            max_epochs=600, damp_armijo=0.5, gamma_armijo=0.001, damp_momentum=0.5,
            **kwargs):

        sgd_variants = ("SGD-Fixed", "SGD-Decreasing", "SGDM",
                   "SGD-Armijo", "MSL-SGDM-C", "MSL-SGDM-R")

        # dataset = (X_train, y_train, X_test, y_test)
        X_train = dataset[0]
        y_train = dataset[1]
        X_test = dataset[2]
        y_test = dataset[3]

        w0 = (0.5 + 0.5) * np.random.default_rng(42).random(X_train.shape[1] - 1) - 0.5
        w0 = np.insert(w0, 0, 0)  # null bias

        if self.solver in ("L-BFGS-B", "CG"):
            model = minimize(linear, w0, args=(X_train, y_train, self.C),
                             method=self.solver, jac=linear_der, bounds=None)

        elif self.solver == "Newton-CG":
            model = minimize(linear, w0, args=(X_train, y_train, self.C),
                             method=self.solver, jac=linear_der, hess=linear_hess,
                             bounds=None)

        elif self.solver in sgd_variants:
            model = minibatch_gd(w0, X_train, y_train, self.C, batch_size,
                                 step_size, momentum, max_epochs, self.solver,
                                 stop, damp_armijo, gamma_armijo, damp_momentum,
                                 linear, linear_der, f_and_df_linear)

        self.opt_result = model
        self.coef_ = model.x
        self.fun = model.fun
        self.grad = np.linalg.norm(model.jac)

        self.mse = mean_squared_error(y_test, self.predict(X_test))

        return self


    def predict(self, X):
        y_pred = np.dot(X, self.coef_)

        return y_pred


    def __str__(self):

        return (f"Solver: {self.solver}" +
                # f"\nTrain score: {self.metrics_train[0]}" +
                f"MSE: {self.mse:.6f}" +
                f"\nObjective function: {self.fun:.6f}" +
                f"\nGrad norm: {self.grad:.6f}" +
                f"\nSol norm: {np.linalg.norm(self.coef_):.6f}" +
                f"\nRun-time (seconds): {self.opt_result.runtime:.6f}" +
                f"\nEpochs: {self.opt_result.nit}")
