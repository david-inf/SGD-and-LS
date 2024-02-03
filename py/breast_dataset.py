# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 12:18:47 2024

@author: Utente
"""

#%% Packages

import numpy as np
import pandas as pd
from solvers import(l_bfgs_b, minibatch_gd_fixed, minibatch_gd_decreasing,
            minibatch_gd_armijo, minibatch_gdm_fixed, msl_sgdm_c, msl_sgdm_r)
from ml_utils import set_accuracy, optim_data, plot_loss, plot_accuracy, plot_runtime

#%% Breast cancer dataset

# load with constant column
X_train = pd.read_csv("datasets/breast_cancer/breast_X_train.csv").values
y_train = pd.read_csv("datasets/breast_cancer/breast_y_train.csv").values.reshape(-1)
X_test = pd.read_csv("datasets/breast_cancer/breast_X_test.csv").values
y_test = pd.read_csv("datasets/breast_cancer/breast_y_test.csv").values.reshape(-1)

X_train = np.hstack((np.ones((X_train.shape[0],1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0],1)), X_test))

# intercept initial guess already added
w0 = (5 + 5) * np.random.default_rng(42).random(10) - 5

#%% L-BFGS-B

# Train dataset
model1 = l_bfgs_b(w0, X_train, y_train)

# Accuracy
set_accuracy(model1, X_train, y_train, X_test, y_test)

#%% [1-2] SGD Fixed-Decreasing

models_1 = []

sgd_fixed_1 = minibatch_gd_fixed(w0, 0.1, 16, X_train, y_train)
set_accuracy(sgd_fixed_1, X_train, y_train, X_test, y_test)
models_1.append(sgd_fixed_1)

sgd_fixed_2 = minibatch_gd_fixed(w0, 0.25, 16, X_train, y_train)
set_accuracy(sgd_fixed_2, X_train, y_train, X_test, y_test)
models_1.append(sgd_fixed_2)

sgd_fixed_3 = minibatch_gd_fixed(w0, 0.05, 16, X_train, y_train)
set_accuracy(sgd_fixed_3, X_train, y_train, X_test, y_test)
models_1.append(sgd_fixed_3)

sgd_decre_1 = minibatch_gd_decreasing(w0, 1, 16, X_train, y_train)
set_accuracy(sgd_decre_1, X_train, y_train, X_test, y_test)
models_1.append(sgd_decre_1)

sgd_decre_2 = minibatch_gd_decreasing(w0, 0.1, 16, X_train, y_train)
set_accuracy(sgd_decre_2, X_train, y_train, X_test, y_test)
models_1.append(sgd_decre_2)

sgd_decre_3 = minibatch_gd_decreasing(w0, 0.5, 16, X_train, y_train)
set_accuracy(sgd_decre_3, X_train, y_train, X_test, y_test)
models_1.append(sgd_decre_3)

sgd_data_1 = optim_data(models_1)

plot_loss(models_1, [f"model.solver({model.step_size})" for model in models_1])

plot_accuracy(models_1, [f"model.solver({model.step_size})" for model in models_1])

plot_runtime(models_1, [f"model.solver({model.step_size})" for model in models_1])
