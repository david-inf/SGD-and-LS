
#%% Packages
# import time
# import numpy as np
# import sys
# import matplotlib.pyplot as plt
# import pandas as pd

from load_datasets import load_diabetes, load_mg, load_mushrooms, load_phishing
from models import LogisticRegression, LinearRegression
from ml_utils import (optim_data, run_bench, optim_bench,
                      models_summary, diagnostic_epochs, diagnostic_time,
                      plot_loss_time, plot_loss_epochs, diagnostic)
from grid_search import run_solvers, run_solvers_par, grid_search_seq, grid_search_par
# from solvers_utils import f_and_df, logistic, logistic_der

# %% Diabetes

data_diab = load_phishing()

# CDiab = 0.5
# MDiab = 64
# kDiab = 200

benchDiab = run_bench(data_diab, 0.5)
benchDiab_data = optim_bench(benchDiab)

# %% Grid search

# sgdfixed_opt1, _ = grid_search_seq("SGD-Fixed", 0.5, data_diab, (64, 128), (0.1, 0.01, 0.001, 0.0001))
# print("% ----- %")
# sgdfixed_opt2, _ = grid_search_par("SGD-Fixed", 0.5, data_diab, (64, 128), (0.1, 0.01, 0.001, 0.0001))
# sgdfixed_opt2 = grid_search_parallel("SGD-Fixed", 0.5, data_diab, [64, 128])

# sgdm_opt, _ = grid_search_seq("SGDM", 0.5, data_diab, (64, 128))

armijo_opt, _ = grid_search_seq("SGD-Armijo", 0.5, data_diab, (128,), (1, 0.1, 0.01), delta_a=(0.5, 0.9))
print("% ----- %")
armijo_opt, _ = grid_search_par("SGD-Armijo", 0.5, data_diab, (128,), (1, 0.1, 0.01), delta_a=(0.5, 0.9))

# %% Run 3 solvers

# sgdfixed_diab = run_solvers("SGD-Fixed", 0.5, data_diab, 128)
sgdarmijo_diab = run_solvers("SGD-Armijo", 0.5, data_diab, 128)

# %%% SGD-Fixed

# fixed1 = LogisticRegression("SGD-Fixed", C=CDiab)
# fixed1.fit(dataset=data_diab, max_epochs=200, batch_size=64, step_size=0.01)
# print(fixed1)

fixed2 = LogisticRegression("SGD-Fixed", 0.5)
fixed2.fit(dataset=data_diab, max_epochs=200, batch_size=64, step_size=0.01, parallel=True)
print(fixed2)

# sgdfixed_diab = run_solvers("SGD-Fixed", CDiab, data_diab, kDiab, MDiab, (0.5, 0.1, 0.01))

# %%% SGD-Decreasing

decre1 = LogisticRegression("SGD-Decreasing", C=CDiab)
decre1.fit(dataset=data_diab, max_epochs=200, batch_size=64, step_size=1)
print(decre1)

# decre2 = LogisticRegression("SGD-Decreasing", C=CDiab)
# decre2.fit(dataset=data_diab, max_epochs=200, batch_size=64, step_size=1, parallel=True)
# print(decre2)

# sgddecre_diab = run_solvers("SGD-Decreasing", CDiab, data_diab, kDiab, MDiab, (1, 0.1, 0.01))

# %%% SGDM

sgdm1 = LogisticRegression("SGDM", C=CDiab)
sgdm1.fit(dataset=data_diab, max_epochs=200, batch_size=64, step_size=0.1, momentum=0.9)
print(sgdm1)

# sgdm_diab = run_solvers("SGDM", CDiab, data_diab, kDiab, MDiab, (1, 0.1, 0.01), momentum=(0.9, 0.9, 0.9))

# %%% SGD-Armijo

armijo1 = LogisticRegression("SGD-Armijo", C=CDiab)
armijo1.fit(dataset=data_diab, max_epochs=200, batch_size=64, step_size=1)
print(armijo1)

# sgdarmijo_diab = run_solvers("SGD-Armijo", CDiab, data_diab, kDiab, MDiab, (1, 0.1, 0.01))

# %%% MSL-SGDM-C

mslc1 = LogisticRegression("MSL-SGDM-C", C=CDiab)
mslc1.fit(dataset=data_diab, max_epochs=200, batch_size=64, step_size=1)
print(mslc1)

# mslc_diab = run_solvers("MSL-SGDM-C", CDiab, data_diab, kDiab, MDiab, (1, 0.1, 0.01), momentum=(0.9, 0.9, 0.9))

# %%% MSL-SGDM-R

mslr1 = LogisticRegression("MSL-SGDM-R", C=CDiab)
mslr1.fit(dataset=data_diab, max_epochs=200, batch_size=64, step_size=1)
print(mslr1)

# mslr_diab = run_solvers("MSL-SGDM-R", CDiab, data_diab, kDiab, MDiab, step_size=(1, 0.1, 0.01), momentum=(0.9, 0.9, 0.9))

# %%% Diagnostic

# diagnostic_epochs(
#     optim_data(sgdfixed_diab + sgdarmijo_diab),
#     optim_data(sgddecre_diab + sgdarmijo_diab),
#     optim_data(sgdm_diab + mslc_diab),
#     optim_data(sgdm_diab + mslr_diab),
#     benchDiab[0])

# diagnostic_time(
#     optim_data(sgdfixed_diab + sgdarmijo_diab),
#     optim_data(sgddecre_diab + sgdarmijo_diab),
#     optim_data(sgdm_diab + mslc_diab),
#     optim_data(sgdm_diab + mslr_diab),
#     benchDiab[0])

# models_diab = optim_data(sgdfixed_diab + sgddecre_diab + sgdm_diab + sgdarmijo_diab +
#                          mslc_diab + mslr_diab)

# diagnostic(
#     optim_data(sgdfixed_diab + sgdarmijo_diab),
#     optim_data(sgddecre_diab + sgdarmijo_diab),
#     optim_data(sgdm_diab + mslc_diab),
#     optim_data(sgdm_diab + mslr_diab),
#     benchDiab[0])


# fig, axs = plt.subplots(2, 2, layout="constrained", sharey=True, sharex=True,
#                         figsize=(6.4, 4.8))

# plot_loss_time(axs[0,0], optim_data(sgdfixed_diab), scalexy=("log", "log"))

# diagnostic_epochs(
#     optim_data([sgdDiab_fixed1, sgdDiab_fixed2, sgdDiab_fixed3, sgdDiab_armijo1, sgdDiab_armijo2, sgdDiab_armijo3]),
#     optim_data([sgdDiab_decre1, sgdDiab_decre2, sgdDiab_decre3, sgdDiab_armijo1, sgdDiab_armijo2, sgdDiab_armijo3]),
#     optim_data([sgdmDiab1, sgdmDiab2, sgdmDiab3, mslcDiab1, mslcDiab2, mslcDiab3]),
#     optim_data([sgdmDiab1, sgdmDiab2, sgdmDiab3, mslrDiab1, mslrDiab2, mslrDiab3]),
#     benchDiab1)

# diagnostic_time(
#     optim_data([sgdDiab_fixed1, sgdDiab_fixed2, sgdDiab_fixed3, sgdDiab_armijo1, sgdDiab_armijo2, sgdDiab_armijo3]),
#     optim_data([sgdDiab_decre1, sgdDiab_decre2, sgdDiab_decre3, sgdDiab_armijo1, sgdDiab_armijo2, sgdDiab_armijo3]),
#     optim_data([sgdmDiab1, sgdmDiab2, sgdmDiab3, mslcDiab1, mslcDiab2, mslcDiab3]),
#     optim_data([sgdmDiab1, sgdmDiab2, sgdmDiab3, mslrDiab1, mslrDiab2, mslrDiab3]),
#     benchDiab1)

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


# %% Linear Regression

# data_mg = load_mg()

# %% SGD-Fixed

# linearbench1 = LinearRegression().fit(data_mg)
# linearbench2 = LinearRegression("CG").fit(data_mg)
# linearbench3 = LinearRegression("Newton-CG").fit(data_mg)

# linearmodel1 = LinearRegression("SGD-Fixed").fit(data_mg, step_size=0.1, stop=1)
