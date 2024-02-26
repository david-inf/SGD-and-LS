
#%% Packages
# import time
# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from load_datasets import load_diabetes
from models import LogisticRegression
from ml_utils import (run_solvers, optim_data, run_bench, optim_bench,
                      models_summary, diagnostic_epochs, diagnostic_time,
                      plot_loss_time, plot_loss_epochs)
# from solvers_utils import f_and_df, logistic, logistic_der

# %% Diabetes

data_diab = load_diabetes()

CDiab = 1
MDiab = 8
kDiab = 200

benchDiab = run_bench(data_diab, CDiab)
benchDiab_data = optim_bench(benchDiab)

# %%% SGD-Fixed

sgdfixed_diab = run_solvers("SGD-Fixed", CDiab, data_diab, kDiab, MDiab, (0.5, 0.01, 0.001))

# %%% SGD-Decreasing

sgdDiab_decre1 = LogisticRegression("SGD-Decreasing", C=CDiab)
sgdDiab_decre1.fit(dataset=data_diab, max_epochs=kDiab, batch_size=MDiab, step_size=1)

sgdDiab_decre2 = LogisticRegression("SGD-Decreasing", C=CDiab)
sgdDiab_decre2.fit(dataset=data_diab, max_epochs=kDiab, batch_size=MDiab, step_size=0.1)

sgdDiab_decre3 = LogisticRegression("SGD-Decreasing", C=CDiab)
sgdDiab_decre3.fit(dataset=data_diab, max_epochs=kDiab, batch_size=MDiab, step_size=0.01)

# %%% SGDM

sgdmDiab1 = LogisticRegression("SGDM", C=CDiab)
sgdmDiab1.fit(dataset=data_diab, max_epochs=kDiab, batch_size=MDiab, step_size=1, momentum=0.9)

sgdmDiab2 = LogisticRegression("SGDM", C=CDiab)
sgdmDiab2.fit(dataset=data_diab, max_epochs=kDiab, batch_size=MDiab, step_size=0.1, momentum=0.9)

sgdmDiab3 = LogisticRegression("SGDM", C=CDiab)
sgdmDiab3.fit(dataset=data_diab, max_epochs=kDiab, batch_size=MDiab, step_size=0.01, momentum=0.9)

# %%% SGD-Armijo

sgdDiab_armijo1 = LogisticRegression("SGD-Armijo", C=CDiab)
sgdDiab_armijo1.fit(dataset=data_diab, max_epochs=kDiab, batch_size=MDiab, step_size=1)

sgdDiab_armijo2 = LogisticRegression("SGD-Armijo", C=CDiab)
sgdDiab_armijo2.fit(dataset=data_diab, max_epochs=kDiab, batch_size=MDiab, step_size=0.1)

sgdDiab_armijo3 = LogisticRegression("SGD-Armijo", C=CDiab)
sgdDiab_armijo3.fit(dataset=data_diab, max_epochs=kDiab, batch_size=MDiab, step_size=0.01)

# %%% MSL-SGDM-C

mslcDiab1 = LogisticRegression("MSL-SGDM-C", C=CDiab)
mslcDiab1.fit(dataset=data_diab, max_epochs=kDiab, batch_size=MDiab, step_size=1, momentum=0.9)

mslcDiab2 = LogisticRegression("MSL-SGDM-C", C=CDiab)
mslcDiab2.fit(dataset=data_diab, max_epochs=kDiab, batch_size=MDiab, step_size=0.1, momentum=0.9)

mslcDiab3 = LogisticRegression("MSL-SGDM-C", C=CDiab)
mslcDiab3.fit(dataset=data_diab, max_epochs=kDiab, batch_size=MDiab, step_size=0.01, momentum=0.9)

# %%% MSL-SGDM-R

mslrDiab1 = LogisticRegression("MSL-SGDM-R", C=CDiab)
mslrDiab1.fit(dataset=data_diab, max_epochs=kDiab, batch_size=MDiab, step_size=1, momentum=0.9)

mslrDiab2 = LogisticRegression(solver="MSL-SGDM-R", C=CDiab)
mslrDiab2.fit(dataset=data_diab, max_epochs=kDiab, batch_size=MDiab, step_size=0.1, momentum=0.9)

mslrDiab3 = LogisticRegression(solver="MSL-SGDM-R", C=CDiab)
mslrDiab3.fit(dataset=data_diab, max_epochs=kDiab, batch_size=MDiab, step_size=0.01, momentum=0.9)

# %%% Diagnostic

modelsDiab_data = optim_data([sgdDiab_fixed1, sgdDiab_fixed2, sgdDiab_fixed3, sgdDiab_decre1, sgdDiab_decre2, sgdDiab_decre3, sgdmDiab1, sgdmDiab2, sgdmDiab3,
                              sgdDiab_armijo1, sgdDiab_armijo2, sgdDiab_armijo3, mslcDiab1, mslcDiab2, mslcDiab3, mslrDiab1, mslrDiab2, mslrDiab3])



diagnostic_epochs(
    optim_data([sgdDiab_fixed1, sgdDiab_fixed2, sgdDiab_fixed3, sgdDiab_armijo1, sgdDiab_armijo2, sgdDiab_armijo3]),
    optim_data([sgdDiab_decre1, sgdDiab_decre2, sgdDiab_decre3, sgdDiab_armijo1, sgdDiab_armijo2, sgdDiab_armijo3]),
    optim_data([sgdmDiab1, sgdmDiab2, sgdmDiab3, mslcDiab1, mslcDiab2, mslcDiab3]),
    optim_data([sgdmDiab1, sgdmDiab2, sgdmDiab3, mslrDiab1, mslrDiab2, mslrDiab3]),
    benchDiab1)

diagnostic_time(
    optim_data([sgdDiab_fixed1, sgdDiab_fixed2, sgdDiab_fixed3, sgdDiab_armijo1, sgdDiab_armijo2, sgdDiab_armijo3]),
    optim_data([sgdDiab_decre1, sgdDiab_decre2, sgdDiab_decre3, sgdDiab_armijo1, sgdDiab_armijo2, sgdDiab_armijo3]),
    optim_data([sgdmDiab1, sgdmDiab2, sgdmDiab3, mslcDiab1, mslcDiab2, mslcDiab3]),
    optim_data([sgdmDiab1, sgdmDiab2, sgdmDiab3, mslrDiab1, mslrDiab2, mslrDiab3]),
    benchDiab1)

# models = [optim_data([sgdDiab_fixed1, sgdDiab_fixed2, sgdDiab_fixed3]),
#           optim_data([sgdDiab_decre1, sgdDiab_decre2, sgdDiab_decre3])]

# fig, axs = plt.subplots(2, 2, sharey=True, layout="constrained", figsize=(6.4, 4.8))
# # i = 0
# # for ax in axs.flat:
# plot_loss_time(axs[0,0], models[0])
# plot_loss_epochs(axs[1,0], models[0])

# plot_loss_time(axs[0,1], models[1])
# plot_loss_epochs(axs[1,1], models[1])
#     # i += 1

# diagnostic_epochs(data1, data2, data3, data4, bench)



#%% Apple quality dataset

# already with constant column
X_train_apple = pd.read_csv("datasets/apple_quality/apple_X_train.csv").values
y_train_apple = pd.read_csv("datasets/apple_quality/apple_y_train.csv").values.reshape(-1)
X_test_apple = pd.read_csv("datasets/apple_quality/apple_X_test.csv").values
y_test_apple = pd.read_csv("datasets/apple_quality/apple_y_test.csv").values.reshape(-1)

#%% Class LogisticRegression

model0_1 = LogisticRegression(solver="L-BFGS", C=1).fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple)
model0_2 = LogisticRegression(solver="Newton-CG", C=1).fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple)
model0_3 = LogisticRegression(solver="CG", C=1).fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple)

bench_data = optim_bench([model0_1, model0_2, model0_3])

#%% SGD-Fixed
k = 100
M1 = 8

model1_1 = LogisticRegression(solver="SGD-Fixed", C=1, epochs=k, minibatch=M1)
model1_1.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.1)
model1_2 = LogisticRegression(solver="SGD-Fixed", C=1, epochs=k, minibatch=M1)
model1_2.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.01)
model1_3 = LogisticRegression(solver="SGD-Fixed", C=1, epochs=k, minibatch=M1)
model1_3.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.001)

# %% SGD-Decreasing

model2_1 = LogisticRegression(solver="SGD-Decreasing", C=1, epochs=k, minibatch=M1)
model2_1.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=1)
model2_2 = LogisticRegression(solver="SGD-Decreasing", C=1, epochs=k, minibatch=M1)
model2_2.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.5)
model2_3 = LogisticRegression(solver="SGD-Decreasing", C=1, epochs=k, minibatch=M1)
model2_3.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.1)

# %% SGDM

model3_1 = LogisticRegression(solver="SGDM", C=1, epochs=k, minibatch=M1)
model3_1.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.1, momentum=0.9)
model3_2 = LogisticRegression(solver="SGDM", C=1, epochs=k, minibatch=M1)
model3_2.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.01, momentum=0.9)
model3_3 = LogisticRegression(solver="SGDM", C=1, epochs=k, minibatch=M1)
model3_3.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.001, momentum=0.9)

# %%

models2_data = optim_data([model1_1, model1_2, model1_3,
                           model2_1, model2_2, model2_3,
                           model3_1, model3_2, model3_3])

# %% SGD-Armijo

model4_1 = LogisticRegression(solver="SGD-Armijo", C=1, epochs=k, minibatch=M1)
model4_1.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=1)
model4_2 = LogisticRegression(solver="SGD-Armijo", C=1, epochs=k, minibatch=M1)
model4_2.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.5)
model4_3 = LogisticRegression(solver="SGD-Armijo", C=1, epochs=k, minibatch=M1)
model4_3.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.1)

# %% MSL-SGDM-C

model5_1 = LogisticRegression(solver="MSL-SGDM-C", C=1, epochs=k, minibatch=M1)
model5_1.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=1, momentum=0.9)
model5_2 = LogisticRegression(solver="MSL-SGDM-C", C=1, epochs=k, minibatch=M1)
model5_2.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.5, momentum=0.9)
model5_3 = LogisticRegression(solver="MSL-SGDM-C", C=1, epochs=k, minibatch=M1)
model5_3.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.1, momentum=0.9)

# %% MSL-SGDM-R

model6_1 = LogisticRegression(solver="MSL-SGDM-R", C=1, epochs=k, minibatch=M1)
model6_1.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=1, momentum=0.9)
model6_2 = LogisticRegression(solver="MSL-SGDM-R", C=1, epochs=k, minibatch=M1)
model6_2.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.5, momentum=0.9)
model6_3 = LogisticRegression(solver="MSL-SGDM-R", C=1, epochs=k, minibatch=M1)
model6_3.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.1, momentum=0.9)

# %% DataFrames

# models = [optim_data([model1_1, model1_2, model4_1, model4_2]),
#           optim_data([model2_1, model2_2, model4_1, model4_2]),
#           optim_data([model3_1, model3_2, model5_1, model5_2]),
#           optim_data([model3_1, model3_2, model6_1, model6_2])]
# fig, axs = plt.subplots(2, 2, sharey=True, layout="constrained", figsize=(6.4, 4.8))
# i = 0
# for ax in axs.flat:
#     plot_loss(ax, models[i])
#     i += 1

# models1 = optim_data([])

models_data = optim_data([model1_1, model1_2, model1_3, model4_1, model4_2, model4_3,
            model2_1, model2_2, model2_3, model4_1, model4_2, model4_3,
            model3_1, model3_2, model3_3, model5_1, model5_2, model5_3,
            model3_1, model3_2, model3_3, model6_1, model6_2, model6_3])

diagnostic(optim_data([model1_1, model1_2, model1_3, model4_1, model4_2, model4_3]),
           optim_data([model2_1, model2_2, model2_3, model4_1, model4_2, model4_3]),
           optim_data([model3_1, model3_2, model3_3, model5_1, model5_2, model5_3]),
           optim_data([model3_1, model3_2, model3_3, model6_1, model6_2, model6_3]),
           model0_1)

all_models = models_summary(models_data, bench_data)
