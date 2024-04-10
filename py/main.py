
#%% Packages
# import time
# import numpy as np
# import sys
import matplotlib.pyplot as plt
# import pandas as pd

from load_datasets import load_diabetes, load_mg, load_mushrooms, load_phishing, load_w1a
from models import LogisticRegression, LinearRegression
from ml_utils import (optim_data, optim_bench,
                      models_summary,
                      plot_loss_time, plot_loss_epochs, diagnostic, plot_loss_epochs)
from grid_search import run_solvers, run_bench, grid_search, cross_val
# from solvers_utils import f_and_df, logistic, logistic_der

# %% Diabetes

data = load_w1a()

C = 0.5
# MDiab = 64
# kDiab = 200

benchDiab = run_bench(load_w1a(), 0.5)
benchDiab_data = optim_bench(benchDiab)

# %% debug line search

armijo1 = LogisticRegression("SGD-Armijo", C)
# armijo1 = LogisticRegression("MSL-SGDM-C", C)
armijo1.fit(data, 32, 1)

# %%

model1 = LogisticRegression("SGD-Fixed").fit(load_w1a(), **dict(batch_size=64))

# %% cross-validation

sgd1_cv = cross_val(LogisticRegression("SGD-Fixed"), load_w1a(), solver_options=dict(batch_size=64))

# %% Grid search

# sgdfixed_opt1, _ = grid_search_seq("SGD-Fixed", 0.5, data_diab, (64, 128), (0.1, 0.01, 0.001, 0.0001))
# print("% ----- %")
sgdfixed_opt2, _ = grid_search("SGD-Fixed", 0.5, load_phishing(), (64, 128), (0.1, 0.01, 0.001, 0.0001))

# sgdm_opt, _ = grid_search_seq("SGDM", 0.5, data_diab, (64, 128))

# armijo_opt, _ = grid_search_seq("SGD-Armijo", 0.5, data_diab, (128,), (1, 0.1, 0.01), delta_a=(0.5, 0.9))
# print("% ----- %")
# armijo_opt, _ = grid_search_par("SGD-Armijo", 0.5, data_diab, (128,), (1, 0.1, 0.01), delta_a=(0.5, 0.9))

# %% Run 3 solvers

M_grid = (32, 64)
delta_a_grid = (0.3, 0.5, 0.7, 0.9)
delta_m_grid = (0.3, 0.5, 0.7)

# SGD-Fixed
sgdfixed_w1a = run_solvers("SGD-Fixed", C, load_w1a(), M_grid)
# SGD-Decreasing
sgddecre_w1a = run_solvers("SGD-Decreasing", C, load_w1a(), M_grid)
# SGDM
sgdm_w1a = run_solvers("SGDM", C, load_w1a(), M_grid)

# SGD-Armijo
sgdarmijo_w1a = run_solvers("SGD-Armijo", C, load_w1a(), M_grid, delta_a=delta_a_grid)
# MSL-SGDM-C
mslc_w1a = run_solvers("MSL-SGDM-C", C, load_w1a(), M_grid, delta_a=delta_a_grid, delta_m=delta_m_grid, n_jobs=6)
# MSL-SGDM-R
mslr_w1a = run_solvers("MSL-SGDM-R", C, load_w1a(), M_grid, delta_a=delta_a_grid)

# %% Adam

# adam_w1a = LogisticRegression("Adam", C)
# adam_w1a.fit(load_w1a(), 32, 0.001)

adam_opt = grid_search("Adam", C, load_w1a(), (64,), (0.01, 0.001, 0.0001))

# %% plot solvers

# diagnostic(
#     optim_data(sgdfixed_w1a + sgdarmijo_w1a),
#     optim_data(sgddecre_w1a + sgdarmijo_w1a),
#     optim_data(sgdm_w1a + mslc_w1a),
#     optim_data(sgdm_w1a + mslr_w1a))

diagnostic([sgdfixed_w1a, sgddecre_w1a, sgdm_w1a, sgdarmijo_w1a, mslc_w1a, mslr_w1a],
           scalexy=("log", "log", "log", "log"))

# fig, axs = plt.subplots(2, 4, layout="constrained", sharey=True, sharex="col",
#                         figsize=(6.4*2, 4.8*1.5))

# for i, ax in enumerate(axs.flat):
#     if i in (0,1,4,5):
        # 1) Train loss against epochs
# plot_loss_epochs(axs[0,0], optim_data([sgdfixed_w1a[2]]), ("log", "log"))
# 
# plot_loss_epochs(ax, data, scalexy)


# %%% SGD-Fixed

# fixed1 = LogisticRegression("SGD-Fixed", C=CDiab)
# fixed1.fit(dataset=data_diab, max_epochs=200, batch_size=64, step_size=0.01)
# print(fixed1)

# fixed2 = LogisticRegression("SGD-Fixed", 0.5)
# fixed2.fit(dataset=data_diab, max_epochs=200, batch_size=64, step_size=0.01, parallel=True)
# print(fixed2)

# sgdfixed_diab = run_solvers("SGD-Fixed", CDiab, data_diab, kDiab, MDiab, (0.5, 0.1, 0.01))

# %%% SGD-Decreasing

# decre1 = LogisticRegression("SGD-Decreasing", C=CDiab)
# decre1.fit(dataset=data_diab, max_epochs=200, batch_size=64, step_size=1)
# print(decre1)

# decre2 = LogisticRegression("SGD-Decreasing", C=CDiab)
# decre2.fit(dataset=data_diab, max_epochs=200, batch_size=64, step_size=1, parallel=True)
# print(decre2)

# sgddecre_diab = run_solvers("SGD-Decreasing", CDiab, data_diab, kDiab, MDiab, (1, 0.1, 0.01))

# %%% SGDM

# sgdm1 = LogisticRegression("SGDM", C=CDiab)
# sgdm1.fit(dataset=data_diab, max_epochs=200, batch_size=64, step_size=0.1, momentum=0.9)
# print(sgdm1)

# sgdm_diab = run_solvers("SGDM", CDiab, data_diab, kDiab, MDiab, (1, 0.1, 0.01), momentum=(0.9, 0.9, 0.9))

# %%% SGD-Armijo

# armijo1 = LogisticRegression("SGD-Armijo", C=CDiab)
# armijo1.fit(dataset=data_diab, max_epochs=200, batch_size=64, step_size=1)
# print(armijo1)

# sgdarmijo_diab = run_solvers("SGD-Armijo", CDiab, data_diab, kDiab, MDiab, (1, 0.1, 0.01))

# %%% MSL-SGDM-C

# mslc1 = LogisticRegression("MSL-SGDM-C", C=CDiab)
# mslc1.fit(dataset=data_diab, max_epochs=200, batch_size=64, step_size=1)
# print(mslc1)

# mslc_diab = run_solvers("MSL-SGDM-C", CDiab, data_diab, kDiab, MDiab, (1, 0.1, 0.01), momentum=(0.9, 0.9, 0.9))

# %%% MSL-SGDM-R

# mslr1 = LogisticRegression("MSL-SGDM-R", C)
# mslr1.fit(dataset=data_diab, max_epochs=200, batch_size=64, step_size=1)
# print(mslr1)

# mslr_diab = run_solvers("MSL-SGDM-R", CDiab, data_diab, kDiab, MDiab, step_size=(1, 0.1, 0.01), momentum=(0.9, 0.9, 0.9))

# %% Linear Regression

# data_mg = load_mg()


# linearbench1 = LinearRegression().fit(data_mg)
# linearbench2 = LinearRegression("CG").fit(data_mg)
# linearbench3 = LinearRegression("Newton-CG").fit(data_mg)

# linearmodel1 = LinearRegression("SGD-Fixed").fit(load_mg())
