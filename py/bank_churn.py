# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 11:56:29 2024

@author: Utente
"""

#%% Packages

import numpy as np
import pandas as pd
from solvers import(l_bfgs_b, minibatch_gd_fixed, minibatch_gd_decreasing,
            minibatch_gd_armijo, minibatch_gdm_fixed, msl_sgdm_c, msl_sgdm_r)
from ml_utils import set_accuracy, optim_data, diagnostic

#%% Bank churn dataset

X_train = pd.read_csv("datasets/bank_churn/bank_X_train.csv").values
y_train = pd.read_csv("datasets/bank_churn/bank_y_train.csv").values.reshape(-1)
X_test = pd.read_csv("datasets/bank_churn/bank_X_test.csv").values
y_test = pd.read_csv("datasets/bank_churn/bank_y_test.csv").values.reshape(-1)

w0 = (1.5 + 1.5) * np.random.default_rng(42).random(18) - 1.5

#%% L-BFGS-B

model1 = l_bfgs_b(w0, X_train, y_train)
set_accuracy(model1, X_train, y_train, X_test, y_test)

#%% [1-2] SGD - Fixed/Decreasing

sgd_fixed_1 = minibatch_gd_fixed(w0, 0.1, 128, X_train, y_train)
set_accuracy(sgd_fixed_1, X_train, y_train, X_test, y_test)
