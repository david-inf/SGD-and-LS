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
# import pandas as pd
from sklearn.model_selection import train_test_split
from myLogisticRegression import myLogRegr

# from scipy.optimize import minimize

# class myLogistic():
    # def __init__(self, ):

#### Breast cancer dataset
# N=569, p=30
from sklearn.datasets import load_breast_cancer

breast = load_breast_cancer()

X, y = breast.data, breast.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

w0 = np.ones(X.shape[1])

model1 = myLogRegr(epochs=3).fit(X_train, y_train, w0)
model2 = myLogRegr(epochs=200, solver="miniGD-decreasing").fit(X_train, y_train, w0)

# M, alpha, lam = 5, 1, 0.5
# compare1 = []
# w_seq1 = miniGD_fixed(X, y, 5, 0.5, np.array([1,1,1,1]), 1)
# w_seq2 = miniGD_fixed(X, y, 5, 0.5, np.array([1,1,1,1]), 0.85)
# w_seq3 = miniGD_fixed(X, y, 5, 0.5, np.array([1,1,1,1]), 0.7)
# w_seq4 = miniGD_fixed(X, y, 5, 0.5, np.array([1,1,1,1]), 0.5)
# w_seq5 = miniGD_fixed(X, y, 5, 0.5, np.array([1,1,1,1]), 0.25)
# w_seq6 = miniGD_fixed(X, y, 5, 0.5, np.array([1,1,1,1]), 0.1)

# fun_seq1 = []
# fun_seq2 = []
# fun_seq3 = []
# fun_seq4 = []
# fun_seq5 = []
# fun_seq6 = []
# for i in range(len(w_seq1)):
#     fun_seq1.append(logistic(X, y, w_seq1[i]))
#     fun_seq2.append(logistic(X, y, w_seq2[i]))
#     fun_seq3.append(logistic(X, y, w_seq3[i]))
#     fun_seq4.append(logistic(X, y, w_seq4[i]))
#     fun_seq5.append(logistic(X, y, w_seq5[i]))
#     fun_seq6.append(logistic(X, y, w_seq6[i]))

# # plt.plot(fun_seq1)
# plt.plot(fun_seq2)
# plt.plot(fun_seq3)
# plt.plot(fun_seq4)
# plt.plot(fun_seq5)
# plt.plot(fun_seq6)
# plt.legend([r"$\alpha=0.85$", r"$\alpha=0.7$", r"$\alpha=0.5$",
#             r"$\alpha=0.25$", r"$\alpha=0.1$"])
# plt.show()
