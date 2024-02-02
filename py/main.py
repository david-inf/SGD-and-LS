# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 21:20:04 2024

@author: Utente
"""

#%% Packages
# import time
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
# from myLogisticRegression import myLogRegr
from solvers import(l_bfgs_b, minibatch_gd_fixed, minibatch_gd_decreasing,
            minibatch_gd_armijo, minibatch_gdm_fixed, msl_sgdm_c, msl_sgdm_r)
from plot_utils import plot_loss, plot_accuracy
from ml_utils import get_accuracy, optim_data

# rng = np.random.default_rng(42)

#%% Apple quality dataset

# load with constant column
X_train_apple = pd.read_csv("datasets/apple_quality/apple_X_train.csv").values
y_train_apple = pd.read_csv("datasets/apple_quality/apple_y_train.csv").values.reshape(-1)
X_test_apple = pd.read_csv("datasets/apple_quality/apple_X_test.csv").values
y_test_apple = pd.read_csv("datasets/apple_quality/apple_y_test.csv").values.reshape(-1)

X_train_apple = np.hstack((np.ones((X_train_apple.shape[0],1)), X_train_apple))
X_test_apple = np.hstack((np.ones((X_test_apple.shape[0],1)), X_test_apple))

# intercept initial guess already added
w0 = np.array([-4, 3, -1, 1, 0, 2, 2.5, -1])

#%% Benchmark solver

# Train dataset
model1 = l_bfgs_b(w0, X_train_apple, y_train_apple)

# Train accuracy
model1.accuracy_train = get_accuracy(model1, X_train_apple, y_train_apple)

# Test accuracy
model1.accuracy_test = get_accuracy(model1, X_test_apple, y_test_apple)

# Results
print("Solver: L-BFGS-B" +
      f"\nTrain accuracy: {model1.accuracy_train:.4f}" +
      f"\nTest accuracy: {model1.accuracy_test:.4f}" +
      f"\nSolution: {model1.x}" +
      f"\nLoss: {(model1.fun / X_train_apple.shape[0]):.4f}" +
      f"\nGradient: {(np.linalg.norm(model1.jac) / X_train_apple.shape[0]):.4f}" +
      "\nMessage: " + model1.message)

#%% Minibatch Gradient Descent with fixed step-size

models1 = []
M1 = 32

sgd_fixed_1 = minibatch_gd_fixed(w0, 0.1, M1, X_train_apple, y_train_apple)
sgd_fixed_1.accuracy_train = get_accuracy(sgd_fixed_1, X_train_apple, y_train_apple)
sgd_fixed_1.accuracy_test = get_accuracy(sgd_fixed_1, X_test_apple, y_test_apple)
models1.append(sgd_fixed_1)

sgd_fixed_2 = minibatch_gd_fixed(w0, 0.01, M1, X_train_apple, y_train_apple)
sgd_fixed_2.accuracy_train = get_accuracy(sgd_fixed_2, X_train_apple, y_train_apple)
sgd_fixed_2.accuracy_test = get_accuracy(sgd_fixed_2, X_test_apple, y_test_apple)
models1.append(sgd_fixed_2)

sgd_fixed_3 = minibatch_gd_fixed(w0, 0.001, M1, X_train_apple, y_train_apple)
sgd_fixed_3.accuracy_train = get_accuracy(sgd_fixed_3, X_train_apple, y_train_apple)
sgd_fixed_3.accuracy_test = get_accuracy(sgd_fixed_3, X_test_apple, y_test_apple)
models1.append(sgd_fixed_3)

sgd_data_1 = optim_data(models1)

plot_loss(models1, [f"MiniGD-fixed({model.step_size})" for model in models1])

plot_accuracy(models1)

#%% Minibatch Gradient Descent with decreasing step-size

models3 = []

model3_1 = minibatch_gd_decreasing(w0, 1, 16, X_train, y_train)
model3_1.accuracy_train = get_accuracy(model3_1, X_train, y_train)
model3_1.accuracy_test = get_accuracy(model3_1, X_test, y_test)
models3.append(model3_1)

model3_2 = minibatch_gd_decreasing(w0, 0.1, 16, X_train, y_train)
model3_2.accuracy_train = get_accuracy(model3_2, X_train, y_train)
model3_2.accuracy_test = get_accuracy(model3_2, X_test, y_test)
models3.append(model3_2)

model3_3 = minibatch_gd_decreasing(w0, 0.01, 16, X_train, y_train)
model3_3.accuracy_train = get_accuracy(model3_3, X_train, y_train)
model3_3.accuracy_test = get_accuracy(model3_3, X_test, y_test)
models3.append(model3_3)

model3_data = optim_data(models3)

plot_loss(models3, [f"alpha={model.initial_step_size}" for model in models3])

#%% SGD Armijo

models_armijo = []

model_armijo_1 = minibatch_gd_armijo(w0, 1, 8, X_train, y_train)
model_armijo_1.accuracy_train = get_accuracy(model_armijo_1, X_train, y_train)
model_armijo_1.accuracy_test = get_accuracy(model_armijo_1, X_test, y_test)
models_armijo.append(model_armijo_1)

model_armijo_2 = minibatch_gd_armijo(w0, 1, 16, X_train, y_train)
model_armijo_2.accuracy_train = get_accuracy(model_armijo_2, X_train, y_train)
model_armijo_2.accuracy_test = get_accuracy(model_armijo_2, X_test, y_test)
models_armijo.append(model_armijo_2)

model_armijo_3 = minibatch_gd_armijo(w0, 1, 32, X_train, y_train)
model_armijo_3.accuracy_train = get_accuracy(model_armijo_3, X_train, y_train)
model_armijo_3.accuracy_test = get_accuracy(model_armijo_3, X_test, y_test)
models_armijo.append(model_armijo_3)

model_armijo_data = optim_data(models_armijo)

plot_loss(models_armijo, [f"M={model.minibatch_size}" for model in models_armijo])

#%% SGDM - compare sensitivity to its step-size

models_sgdm = []

model_sgdm_1 = minibatch_gdm_fixed(w0, 1, 0.9, 128, X_train, y_train)
model_sgdm_1.accuracy_train = get_accuracy(model_sgdm_1, X_train, y_train)
model_sgdm_1.accuracy_test = get_accuracy(model_sgdm_1, X_test, y_test)
models_sgdm.append(model_sgdm_1)

model_sgdm_2 = minibatch_gdm_fixed(w0, 0.1, 0.9, 128, X_train, y_train)
model_sgdm_2.accuracy_train = get_accuracy(model_sgdm_2, X_train, y_train)
model_sgdm_2.accuracy_test = get_accuracy(model_sgdm_2, X_test, y_test)
models_sgdm.append(model_sgdm_2)

model_sgdm_3 = minibatch_gdm_fixed(w0, 0.01, 0.9, 128, X_train, y_train)
model_sgdm_3.accuracy_train = get_accuracy(model_sgdm_3, X_train, y_train)
model_sgdm_3.accuracy_test = get_accuracy(model_sgdm_3, X_test, y_test)
models_sgdm.append(model_sgdm_3)

model_sgdm_data = optim_data(models_sgdm)

plot_loss(models_sgdm, [f"alpha={model.step_size}" for model in models_sgdm])

#%% MSL-SGDM-C

models_msl_c = []

model_msl_c_1 = msl_sgdm_c(w0, 1, 0.8, 32, X_train, y_train)
model_msl_c_1.accuracy_train = get_accuracy(model_msl_c_1, X_train, y_train)
model_msl_c_1.accuracy_test = get_accuracy(model_msl_c_1, X_test, y_test)
models_msl_c.append(model_msl_c_1)

model_msl_c_2 = msl_sgdm_c(w0, 0.1, 0.8, 32, X_train, y_train)
model_msl_c_2.accuracy_train = get_accuracy(model_msl_c_2, X_train, y_train)
model_msl_c_2.accuracy_test = get_accuracy(model_msl_c_2, X_test, y_test)
models_msl_c.append(model_msl_c_2)

model_msl_c_3 = msl_sgdm_c(w0, 0.01, 0.8, 32, X_train, y_train)
model_msl_c_3.accuracy_train = get_accuracy(model_msl_c_3, X_train, y_train)
model_msl_c_3.accuracy_test = get_accuracy(model_msl_c_3, X_test, y_test)
models_msl_c.append(model_msl_c_3)

model_msl_c_data = optim_data(models_msl_c)

plot_loss(models_msl_c, [f"alpha0={model.initial_step_size}" for model in models_msl_c])

#%%

# Apple quality

# al diminuire della dimensione del minibatch si riduce il learning rate e viceversa

# lam = 5e-3
# M = 4
# k = 250

# Train dataset
# start = time.time()
# print(minibatch_gd_fixed(w0, 1e-3, 8, X_train, y_train))
# end = time.time()
# print(f"seconds: {end - start}")

# start = time.time()
# print(minibatch_gd_decreasing(w0, 0.1, 8, X_train, y_train))
# end = time.time()
# print(f"seconds: {end - start}")

# print(solvers.minibatch_gd_fixed(w0, 1e-3, M=8, X=X_train, y=y_train, coeff=5e-3))
# print(solvers.minibatch_gd_decreasing(w0, 0.1, M=8, X=X_train, y=y_train, coeff=5e-3))
# start = time.time()
# print(minibatch_gd_armijo(w0, 1e-10, M=8, X=X_train, y=y_train))
# end = time.time()
# print(f"seconds: {end - start}")
# model2_opt1 = myLogRegr(
#     minibatch_size=M, solver="miniGD-fixed", regul_coef=lam, epochs=k)
# model2_opt1.fit(
#     X_train, y_train, w0, learning_rate=1e-3)

# start = time.time()
# print(msl_sgdm_c(w0, 0.001, 0.8, 8, X_train, y_train))
# end = time.time()
# print(f"seconds: {end - start}")

# start = time.time()
# print(msl_sgdm_r(w0, 0.001, 0.9, 8, X_train, y_train))
# end = time.time()
# print(f"seconds: {end - start}")

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
