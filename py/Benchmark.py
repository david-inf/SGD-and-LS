# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 10:42:30 2024

@author: Utente
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
# from sklearn.metrics import log_loss
# from myLogisticRegression import myLogRegr

rng = np.random.default_rng(20)

appleDF = pd.read_csv("./datasets/apple_quality.csv", nrows=400)
appleDF.drop("A_id", axis=1, inplace=True)
# ReType
appleDF["Acidity"] = appleDF['Acidity'].astype(float)
appleDF['Quality'] = appleDF['Quality'].map({'bad': -1, 'good': 1})

X = appleDF.drop("Quality", axis=1)
y = appleDF["Quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
N = X_train.shape[0]
X_train = np.hstack((np.ones((N,1)),X_train.values))  # add bias
y_train = y_train.values

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic(w, X, y):
    # global X_train, y_train, N
    # X, y = X_train, y_train
    lam = 0.5
    loss = 0  # sum of loss over singles dataset examples
    for i in range(N):
        loss += np.log(1 + np.exp(- y[i] * np.dot(np.transpose(w0), X[i,:])))
    regul = lam * np.linalg.norm(w) ** 2
    return loss + regul

def logistic_der(w, X, y):
    # global X_train, y_train, N
    # X, y = X_train, y_train
    lam = 0.5
    r = np.zeros(N)
    for i in range(N):
        r[i] = - y[i] * sigmoid(- y[i] * np.dot(np.transpose(w), X[i,:]))
    loss_der = np.dot(np.transpose(X), r)
    regul_der = lam * 2 * w
    return loss_der + regul_der

def optimalSolver(fun, grad, w0, X, y):
    res = minimize(fun, w0, args=(X, y), method="L-BFGS-B", jac=grad,
                   bounds=None, options={"disp": True})
    return res

# w0 = (4 - 1) * rng.random(X_train.shape[1]) + 1
# w0 = np.insert(w0, 0, 1)  # add initial guess for intercept
w0 = np.array([-3.43045935, 3.82898442, 0.49743289, 2.82647303, 0.32708793,
                1.9157581, 0.13846527, -2.2005224])
w_star = np.array([-3.93045935,  4.32898442,  0.49743289,  2.82647303,  0.32708793,
        1.9957581 , -0.36153473, -1.5005224])
# w0 = np.zeros(X_train.shape[1])


model = optimalSolver(logistic, logistic_der, w_star, X_train, y_train)























