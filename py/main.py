
#%% Packages
# import time
# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from load_datasets import load_diabetes
from models import LogisticRegression, LinearRegression
from ml_utils import (run_solvers, optim_data, run_bench, optim_bench,
                      models_summary, diagnostic_epochs, diagnostic_time,
                      plot_loss_time, plot_loss_epochs, diagnostic)
# from solvers_utils import f_and_df, logistic, logistic_der

# %% Diabetes

data_diab = load_diabetes()

CDiab = 1
MDiab = 8
kDiab = 200

benchDiab = run_bench(data_diab, CDiab)
benchDiab_data = optim_bench(benchDiab)

# %%% SGD-Fixed

sgdfixed_diab = run_solvers("SGD-Fixed", CDiab, data_diab, kDiab, MDiab, (0.1, 0.01, 0.001))

# %%% SGD-Decreasing

sgddecre_diab = run_solvers("SGD-Decreasing", CDiab, data_diab, kDiab, MDiab, (1, 0.1, 0.01))

# %%% SGDM

sgdm_diab = run_solvers("SGDM", CDiab, data_diab, kDiab, MDiab, (0.1, 0.01, 0.001), momentum=(0.9, 0.9, 0.9))

# %%% SGD-Armijo

sgdarmijo_diab = run_solvers("SGD-Armijo", CDiab, data_diab, kDiab, MDiab, (1, 0.1, 0.01))

# %%% MSL-SGDM-C

mslc_diab = run_solvers("MSL-SGDM-C", CDiab, data_diab, kDiab, MDiab, (1, 0.1, 0.01), momentum=(0.9, 0.9, 0.9))

# %%% MSL-SGDM-R

mslr_diab = run_solvers("MSL-SGDM-R", CDiab, data_diab, kDiab, MDiab, step_size=(1, 0.1, 0.01), momentum=(0.9, 0.9, 0.9))

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

models_diab = optim_data(sgdfixed_diab + sgddecre_diab + sgdm_diab + sgdarmijo_diab +
                         mslc_diab + mslr_diab)

diagnostic(
    optim_data(sgdfixed_diab + sgdarmijo_diab),
    optim_data(sgddecre_diab + sgdarmijo_diab),
    optim_data(sgdm_diab + mslc_diab),
    optim_data(sgdm_diab + mslr_diab),
    benchDiab[0])


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

linearbench1 = LinearRegression().fit(data_diab)
