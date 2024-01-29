# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 21:20:04 2024

@author: Utente
"""

#%% Packages
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from myLogisticRegression import myLogRegr
from myUtils import plotDiagnostic

rng = np.random.default_rng(42)

#%% Apple quality dataset

appleDF = pd.read_csv("./datasets/apple_quality.csv", nrows=400)
appleDF = appleDF.drop(columns={"A_id"})
# ReType
appleDF["Acidity"] = appleDF['Acidity'].astype(float)
appleDF['Quality'] = appleDF['Quality'].map({'bad': -1, 'good': 1})

X = appleDF.drop("Quality", axis=1)
y = appleDF["Quality"]

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state = 42)
X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values

w0 = np.array([-4, 3, -1, 1, 0, 2, 2.5, -1])

#%%% Benchmark solver

#%%% miniGD-fixed


#%% Benchmark solver

# Apple quality

model1_opt = myLogRegr(solver="scipy", regul_coef=0.1)
model1_opt.fit(X_train, y_train, w0)

model1_pred = model1_opt.predict(X_test)
model1_accuracy = accuracy_score(y_test, model1_pred)
print(f"{model1_opt.solver} accuracy: {model1_accuracy:.6f}" +
      "\nSolution: {model1_opt.coef_}" + f"\nLoss: {model1_opt.obj_}" +
      "\nGradient: {model1_opt.grad_}" +
      "\nSolver message: " + model1_opt.solver_message)

#%% miniGD-fixed

# Apple quality

# al diminuire della dimensione del minibatch si riduce il learning rate e viceversa

lam = 0.05
M = 4
k = 700

model2_opt1 = myLogRegr(
    minibatch_size=M, solver="miniGD-fixed", regul_coef=lam, epochs=k)
model2_opt1.fit(
    X_train, y_train, w0, learning_rate=1e-3)

model2_pred1 = model2_opt1.predict(X_test)
model2_accuracy1 = accuracy_score(y_test, model2_pred1)
print(f"{model2_accuracy1:.6f}")


model2_opt2 = myLogRegr(
    minibatch_size=M, solver="miniGD-fixed", regul_coef=lam, epochs=k)
model2_opt2.fit(
    X_train, y_train, w0, learning_rate=5e-4)

model2_pred2 = model2_opt2.predict(X_test)
model2_accuracy2 = accuracy_score(y_test, model2_pred2)
print(f"{model2_accuracy2:.6f}")


model2_opt3 = myLogRegr(
    minibatch_size=M, solver="miniGD-fixed", regul_coef=lam, epochs=k)
model2_opt3.fit(
    X_train, y_train, w0, learning_rate=1e-5)

model2_pred3 = model2_opt3.predict(X_test)
model2_accuracy3 = accuracy_score(y_test, model2_pred3)
print(f"{model2_accuracy3:.6f}")


model2_opt4 = myLogRegr(
    minibatch_size=M, solver="miniGD-fixed", regul_coef=lam, epochs=k)
model2_opt4.fit(
    X_train, y_train, w0, learning_rate=1e-4)

model2_pred4 = model2_opt4.predict(X_test)
model2_accuracy4 = accuracy_score(y_test, model2_pred4)
print(f"{model2_accuracy4:.6f}")


plotDiagnostic(
    [model2_opt1, model2_opt2, model2_opt3, model2_opt4],
    [r"$\alpha=1e-3$", r"$\alpha=5e-4$", r"$\alpha=1e-5$", r"$\alpha=1e-4$"])

#%% miniGD-decreasing

# Apple quality

model3_opt = myLogRegr()
model3_opt.fit(X_train, y_train, w0)

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





