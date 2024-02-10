
#%% Packages
# import time
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
# from myLogisticRegression import myLogRegr
# from solvers import(l_bfgs_b, sgd_fixed, sgd_decreasing, sgd_armijo,
#                     sgdm, msl_sgdm_c, msl_sgdm_r)
from ml_utils import set_accuracy, optim_data, diagnostic, optim_bench

from models import LogisticRegression

#%% Apple quality dataset

# load with constant column
X_train_apple = pd.read_csv("datasets/apple_quality/apple_X_train.csv").values
y_train_apple = pd.read_csv("datasets/apple_quality/apple_y_train.csv").values.reshape(-1)
X_test_apple = pd.read_csv("datasets/apple_quality/apple_X_test.csv").values
y_test_apple = pd.read_csv("datasets/apple_quality/apple_y_test.csv").values.reshape(-1)

# intercept initial guess already added
# w0 = np.array([-4, 3, -1, 1, 0, 2, 2.5, -1])
rng = np.random.default_rng(42)
w0 = (5 + 5) * rng.random(8) - 5

#%% Class LogisticRegression

model0_1 = LogisticRegression().fit(w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple)
model0_2 = LogisticRegression(solver="Newton-CG").fit(w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple)
model0_3 = LogisticRegression(solver="CG").fit(w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple)

#%%

model1_1 = LogisticRegression(solver="SGD-Fixed", C=1)
model1_1.fit(w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.1)

model2_1 = LogisticRegression(solver="SGD-Decreasing", C=1)
model2_1.fit(w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=1)

model3_1 = LogisticRegression(solver="SGDM", C=1)
model3_1.fit(w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.1, momentum=0.9)

model4_1 = LogisticRegression(solver="SGD-Armijo", C=1)
model4_1.fit(w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=1)

model5_1 = LogisticRegression(solver="MSL-SGDM-C", C=1)
model5_1.fit(w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.1, momentum=0.9)

model6_1 = LogisticRegression(solver="MSL-SGDM-R", C=1)
model6_1.fit(w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.1, momentum=0.9)


models1_data = optim_data([model1_1, model2_1, model3_1, model4_1, model5_1, model6_1])
models1_bench = optim_bench([model0_1, model0_2, model0_3])

diagnostic(models1_data)



# model1_2 = LogisticRegression(solver="SGD-Fixed", C=0.5).fit(
#     w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple,
#     step_size=0.01)

# model1_3 = LogisticRegression(solver="SGD-Fixed", C=0.5).fit(
#     w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple,
#     step_size=0.05)

# model2_2 = LogisticRegression(solver="SGD-Decreasing", C=0.5).fit(
#     w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple,
#     step_size=0.01)

# model2_3 = LogisticRegression(solver="SGD-Decreasing", C=0.5).fit(
#     w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple,
#     step_size=0.05)

# model3 = LogisticRegression(solver="SGD-Armijo", C=0.5).fit(
#     w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple,
#     step_size=0.01)

# model4 = LogisticRegression(solver="SGDM", C=0.1).fit(
#     w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple,
#     step_size=0.1, momentum=0.8)

# model5 = LogisticRegression(solver="MSL-SGDM-C", C=0.1).fit(
#     w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple,
#     step_size=0.1, momentum=0.9)

# model6 = LogisticRegression(solver="MSL-SGDM-R", C=0.1).fit(
#     w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple,
#     step_size=0.1, momentum=0.9)

# diagnostic_plots([model1, model2, model3])

#%% Benchmark solver

model1 = l_bfgs_b(w0, X_train_apple, y_train_apple)
set_accuracy(model1, X_train_apple, y_train_apple, X_test_apple, y_test_apple)

#%% [1-2] SGD Fixed-Decreasing

models1 = []
M1 = 32

sgd_fixed_1 = sgd_fixed(w0, 0.1, M1, X_train_apple, y_train_apple)
set_accuracy(sgd_fixed_1, X_train_apple, y_train_apple, X_test_apple, y_test_apple)
models1.append(sgd_fixed_1)

sgd_fixed_2 = sgd_fixed(w0, 0.01, M1, X_train_apple, y_train_apple)
set_accuracy(sgd_fixed_2, X_train_apple, y_train_apple, X_test_apple, y_test_apple)
models1.append(sgd_fixed_2)

sgd_fixed_3 = sgd_fixed(w0, 0.001, M1, X_train_apple, y_train_apple)
set_accuracy(sgd_fixed_3, X_train_apple, y_train_apple, X_test_apple, y_test_apple)
models1.append(sgd_fixed_3)

sgd_decre_1 = sgd_decreasing(w0, 1, M1, X_train_apple, y_train_apple)
set_accuracy(sgd_decre_1, X_train_apple, y_train_apple, X_test_apple, y_test_apple)
models1.append(sgd_decre_1)

sgd_decre_2 = sgd_decreasing(w0, 0.1, M1, X_train_apple, y_train_apple)
set_accuracy(sgd_decre_2, X_train_apple, y_train_apple, X_test_apple, y_test_apple)
models1.append(sgd_decre_2)

sgd_decre_3 = sgd_decreasing(w0, 0.05, M1, X_train_apple, y_train_apple)
set_accuracy(sgd_decre_3, X_train_apple, y_train_apple, X_test_apple, y_test_apple)
models1.append(sgd_decre_3)

# sgd_data_1 = optim_data(models1)

diagnostic(models1,
    [f"{model.solver}({model.step_size})" for model in models1], start_loss=25)

#%% [3] SGD Armijo

models_armijo = []

sgd_armijo_1 = minibatch_gd_armijo(w0, 1, 32, X_train_apple, y_train_apple)
set_accuracy(sgd_armijo_1, X_train_apple, y_train_apple, X_test_apple, y_test_apple)
models_armijo.append(sgd_armijo_1)

sgd_armijo_2 = minibatch_gd_armijo(w0, 0.1, 32, X_train_apple, y_train_apple)
set_accuracy(sgd_armijo_2, X_train_apple, y_train_apple, X_test_apple, y_test_apple)
models_armijo.append(sgd_armijo_2)

sgd_armijo_3 = minibatch_gd_armijo(w0, 0.001, 32, X_train_apple, y_train_apple)
set_accuracy(sgd_armijo_3, X_train_apple, y_train_apple, X_test_apple, y_test_apple)
models_armijo.append(sgd_armijo_3)

model_armijo_data = optim_data(models_armijo)

diagnostic(models_armijo, [f"{model.solver}({model.step_size})" for model in models_armijo])

#%% [4] SGDM - compare sensitivity to its step-size

models_sgdm = []

sgdm_1 = minibatch_gdm_fixed(w0, 1, 0.9, 128, X_train_apple, y_train_apple)
set_accuracy(sgdm_1, X_train_apple, y_train_apple, X_test_apple, y_test_apple)
models_sgdm.append(sgdm_1)

sgdm_2 = minibatch_gdm_fixed(w0, 0.2, 0.9, 128, X_train_apple, y_train_apple)
set_accuracy(sgdm_2, X_train_apple, y_train_apple, X_test_apple, y_test_apple)
models_sgdm.append(sgdm_2)

sgdm_3 = minibatch_gdm_fixed(w0, 0.5, 0.9, 128, X_train_apple, y_train_apple)
set_accuracy(sgdm_3, X_train_apple, y_train_apple, X_test_apple, y_test_apple)
models_sgdm.append(sgdm_3)

model_sgdm_data = optim_data(models_sgdm)

diagnostic(models_sgdm, [f"{model.solver}({model.step_size})" for model in models_sgdm])

#%% [5] MSL-SGDM-C/R - step-size

models_msl = []

msl_c_1 = msl_sgdm_c(w0, 1, 0.9, 128, X_train_apple, y_train_apple)
set_accuracy(msl_c_1, X_train_apple, y_train_apple, X_test_apple, y_test_apple)
models_msl.append(msl_c_1)

msl_c_2 = msl_sgdm_c(w0, 0.1, 0.9, 128, X_train_apple, y_train_apple)
set_accuracy(msl_c_2, X_train_apple, y_train_apple, X_test_apple, y_test_apple)
models_msl.append(msl_c_2)

msl_c_3 = msl_sgdm_c(w0, 0.2, 0.9, 128, X_train_apple, y_train_apple)
set_accuracy(msl_c_3, X_train_apple, y_train_apple, X_test_apple, y_test_apple)
models_msl.append(msl_c_3)

msl_r_1 = msl_sgdm_r(w0, 1, 0.8, 128, X_train_apple, y_train_apple)
set_accuracy(msl_r_1, X_train_apple, y_train_apple, X_test_apple, y_test_apple)
models_msl.append(msl_r_1)

msl_r_2 = msl_sgdm_r(w0, 0.1, 0.8, 128, X_train_apple, y_train_apple)
set_accuracy(msl_r_2, X_train_apple, y_train_apple, X_test_apple, y_test_apple)
models_msl.append(msl_r_2)

msl_r_3 = msl_sgdm_r(w0, 0.01, 0.8, 128, X_train_apple, y_train_apple)
set_accuracy(msl_r_3, X_train_apple, y_train_apple, X_test_apple, y_test_apple)
models_msl.append(msl_r_3)

model_msl_data = optim_data(models_msl)

diagnostic(models_msl, [f"{model.solver}({model.step_size})" for model in models_msl])

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
