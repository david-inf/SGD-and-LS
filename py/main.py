# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 21:20:04 2024

@author: Utente
"""

import numpy as np
# import matplotlib.pyplot as plt
# np.linalg.norm()
# import scipy.linalg as la
# la.norm()
import pandas as pd
from sklearn.model_selection import train_test_split
from myLogisticRegression import myLogRegr

rng = np.random.default_rng(42)
# from scipy.optimize import minimize

# class myLogistic():
    # def __init__(self, ):

#### Breast cancer dataset
# N=569, p=30
# from sklearn.datasets import load_breast_cancer

# breast = load_breast_cancer()

# X, y = breast.data, breast.target
# y[y == 0] = -1
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# w0 = np.ones(X.shape[1])

# w0 = (4 - 1) * rng.random(X.shape[1]) + 1

# model1 = myLogRegr(6)
# model1.fit(X_train, y_train, w0)
# model1.plotDiagnostic()

# model2 = myLogRegr(12, epochs=100)
# model2.fit(X_train, y_train, w0, 0.5)
# model2.plotDiagnostic()

# model3 = myLogRegr(8, epochs=100, solver="miniGD-decreasing")
# model3.fit(X_train, y_train, w0, alpha=10)
# model3.plotDiagnostic()

# model4 = myLogRegr(6, solver="miniGD-armijo")
# model4.fit(X_train, y_train, w0)
# model4.plotDiagnostic()

#### Apple quality

appleDF = pd.read_csv("./datasets/apple_quality.csv", nrows=400)
appleDF.drop("A_id", axis=1, inplace=True)
# ReType
appleDF["Acidity"] = appleDF['Acidity'].astype(float)
appleDF['Quality'] = appleDF['Quality'].map({'bad': -1, 'good': 1})

X = appleDF.drop("Quality", axis=1)
y = appleDF["Quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
X_train = X_train.values
y_train = y_train.values

w0 = (4 - 1) * rng.random(X_train.shape[1]) + 1

# model1 = myLogRegr(minibatch_size=5, epochs=20)
# model1.fit(X_train, y_train, w0, alpha=0.9)
# model1.plotDiagnostic()

# model2 = myLogRegr(minibatch_size=5, epochs=20)
# model2.fit(X_train, y_train, w0, alpha=0.8)
# model2.plotDiagnostic()

# model3 = myLogRegr(minibatch_size=5, epochs=20)
# model3.fit(X_train, y_train, w0, alpha=0.65)
# model3.plotDiagnostic()

# model4 = myLogRegr(minibatch_size=5, epochs=20)
# model4.fit(X_train, y_train, w0, alpha=0.5)
# model4.plotDiagnostic()

# model5 = myLogRegr(minibatch_size=5, epochs=20)
# model5.fit(X_train, y_train, w0, alpha=0.3)
# model5.plotDiagnostic()

# model_benchmark = myLogRegr(solver="scipy")
# model_benchmark.fit(X_train, y_train, w0)





