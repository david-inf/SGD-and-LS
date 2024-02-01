# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 21:20:04 2024

@author: Utente
"""

#%% Packages
import time
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
# from myLogisticRegression import myLogRegr
# from myUtils import plotDiagnostic
from solvers import(l_bfgs_b, minibatch_gd_fixed, minibatch_gd_decreasing,
            minibatch_gd_armijo, minibatch_gdm_fixed)
from my_utils import predict

# rng = np.random.default_rng(42)

#%% Apple quality dataset

# load with constant column
X_train = pd.read_csv("datasets/apple_quality/apple_X_train.csv").values
y_train = pd.read_csv("datasets/apple_quality/apple_y_train.csv").values.reshape(-1)
X_test = pd.read_csv("datasets/apple_quality/apple_X_test.csv").values
y_test = pd.read_csv("datasets/apple_quality/apple_y_test.csv").values.reshape(-1)

X_train = np.hstack((np.ones((X_train.shape[0],1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0],1)), X_test))

# intercept initial guess already added
w0 = np.array([-4, 3, -1, 1, 0, 2, 2.5, -1])

#%% Benchmark solver

# Apple quality

# Train dataset
# start = time.time()
model1 = l_bfgs_b(w0, X_train, y_train)
# end = time.time()
# print(f"seconds: {end - start}")
# model1_opt = myLogRegr(
#     solver="scipy", regul_coef=0.1)
# model1_opt.fit(
#     X_train, y_train, w0)

# Train accuracy
model1_accuracy_train = accuracy_score(y_train, predict(X_train, model1.x))
# model1_accuracy_train = model1_opt.getAccuracy(X_train, y_train)

# Test accuracy
model1_accuracy_test = accuracy_score(y_test, predict(X_test, model1.x))
# model1_accuracy_test = model1_opt.getAccuracy(X_test, y_test)

# Results
# print(f"Solver: {model1_opt.solver}" +
#       f"\nTrain accuracy: {model1_accuracy_train:.4f}" +
#       f"\nTest accuracy: {model1_accuracy_test:.4f}" +
#       f"\nSolution: {model1_opt.coef_}" +
#       f"\nLoss: {model1_opt.obj_:.4f}" +
#       f"\nGradient: {model1_opt.grad_:.8f}" +
#       "\nMessage: " + model1_opt.solver_message)

#%% miniGD-fixed

# Apple quality

# al diminuire della dimensione del minibatch si riduce il learning rate e viceversa

# lam = 5e-3
# M = 4
# k = 250

# Train dataset
start = time.time()
print(minibatch_gd_fixed(w0, 1e-3, 8, X_train, y_train))
end = time.time()
print(f"seconds: {end - start}")

start = time.time()
print(minibatch_gd_decreasing(w0, 0.1, 8, X_train, y_train))
end = time.time()
print(f"seconds: {end - start}")

# print(solvers.minibatch_gd_fixed(w0, 1e-3, M=8, X=X_train, y=y_train, coeff=5e-3))
# print(solvers.minibatch_gd_decreasing(w0, 0.1, M=8, X=X_train, y=y_train, coeff=5e-3))
start = time.time()
print(minibatch_gd_armijo(w0, 1e-10, M=8, X=X_train, y=y_train))
end = time.time()
print(f"seconds: {end - start}")
# model2_opt1 = myLogRegr(
#     minibatch_size=M, solver="miniGD-fixed", regul_coef=lam, epochs=k)
# model2_opt1.fit(
#     X_train, y_train, w0, learning_rate=1e-3)

start = time.time()
print(minibatch_gdm_fixed(w0, 0.001, 0.8, 8, X_train, y_train))
end = time.time()
print(f"seconds: {end - start}")

# Train accuracy
# model2_1_accuracy_train = accuracy_score(y_train, solvers.predict(X_train, ))
# model2_1_accuracy_train = model2_opt1.getAccuracy(X_train, y_train)

# Test accuracy

# model2_1_accuracy_test = model2_opt1.getAccuracy(X_test, y_test)

# Results
# print(f"Solver: {model2_opt1.solver}" +
#       f"\nTrain accuracy: {model2_1_accuracy_train:.4f}" +
#       f"\nTest accuracy: {model2_1_accuracy_test:.4f}" +
#       f"\nSolution: {model2_opt1.coef_}" +
#       f"\nLoss: {model2_opt1.obj_:.4f}" +
#       f"\nGradient: {model2_opt1.grad_:.8f}" +
#       "\nMessage: " + model2_opt1.solver_message)

# plotDiagnostic([model2_opt1], [r"$\alpha=1e-3$"])


# model2_opt2 = myLogRegr(
#     minibatch_size=M, solver="miniGD-fixed", regul_coef=lam, epochs=k)
# model2_opt2.fit(
#     X_train, y_train, w0, learning_rate=5e-4)

# model2_pred2 = model2_opt2.predict(X_test)
# model2_accuracy2 = accuracy_score(y_test, model2_pred2)
# print(f"{model2_accuracy2:.6f}")


# model2_opt3 = myLogRegr(
#     minibatch_size=M, solver="miniGD-fixed", regul_coef=lam, epochs=k)
# model2_opt3.fit(
#     X_train, y_train, w0, learning_rate=1e-5)

# model2_pred3 = model2_opt3.predict(X_test)
# model2_accuracy3 = accuracy_score(y_test, model2_pred3)
# print(f"{model2_accuracy3:.6f}")


# model2_opt4 = myLogRegr(
#     minibatch_size=M, solver="miniGD-fixed", regul_coef=lam, epochs=k)
# model2_opt4.fit(
#     X_train, y_train, w0, learning_rate=1e-4)

# model2_pred4 = model2_opt4.predict(X_test)
# model2_accuracy4 = accuracy_score(y_test, model2_pred4)
# print(f"{model2_accuracy4:.6f}")


# plotDiagnostic(
#     [model2_opt1, model2_opt2, model2_opt3, model2_opt4],
#     [r"$\alpha=1e-3$", r"$\alpha=5e-4$", r"$\alpha=1e-5$", r"$\alpha=1e-4$"])

#%% miniGD-decreasing

# Apple quality

# model3_opt = myLogRegr()
# model3_opt.fit(X_train, y_train, w0)

#%%
### Two-dimensional attempt

# X_train2 = X_train[:,:2]  # two features

# w0_2 = np.array([5, 6, -1])

# model1 = myLogRegr(minibatch_size=4, bias=False, regul_coef=0.1)
# model1.fit(X_train1, y_train, w0_1)

## Optimal solver

# model1_opt = myLogRegr(solver="scipy", regul_coef=0.1)
# model1_opt.fit(X_train2, y_train, w0_2)

# X_test2 = X_test[:,:2]
# y_pred2 = model1_opt.predict(X_test2)
# print(accuracy_score(y_test, y_pred2))

# w0 = (4 - 1) * rng.random(X_train.shape[1]) + 1

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





