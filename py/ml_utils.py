# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def optim_data(models):
    # models: LogisticRegression
    models_data = pd.DataFrame(
        {
            "Solver": [model.solver for model in models],
            "C": [model.C for model in models],
            "Minibatch": [model.opt_result.minibatch_size for model in models],
            "Alpha0": [model.opt_result.step_size for model in models],
            "Beta0": [model.opt_result.momentum for model in models],
            "Solution": [model.coef_ for model in models],
            "l2-Loss": [model.fun for model in models],
            "Grad norm": [model.grad for model in models],
            "Run-time": [model.opt_result.runtime for model in models],
            "Epochs": [model.opt_result.nit for model in models],
            # "Termination": [model.message for model in models],
            "Train score": [model.metrics_train[0] for model in models],
            "Test score": [model.metrics_test[0] for model in models],
            "Bal train score": [model.metrics_train[1] for model in models],
            "Bal test score": [model.metrics_test[1] for model in models],
            "Fun/Epochs": [model.fun_seq for model in models],
            "Time/Epochs": [model.opt_result.time_per_epoch for model in models]
        }
    )
    return models_data


def optim_bench(models):
    data = pd.DataFrame(
        {
            "Solver": [model.solver for model in models],
            "C": [model.C for model in models],
            "Minibatch": np.nan,
            "Alpha0": np.nan,
            "Beta0": np.nan,
            "Solution": [model.coef_ for model in models],
            "l2-Loss": [model.fun for model in models],
            "Grad norm": [model.grad for model in models],
            "Run-time": np.nan,
            "Epochs": [model.opt_result.nit for model in models],
            # "Termination": model.message,
            "Train score": [model.metrics_train[0] for model in models],
            "Test score": [model.metrics_test[0] for model in models],
            "Bal train score": [model.metrics_train[1] for model in models],
            "Bal test score": [model.metrics_test[1] for model in models],
            "Fun/Epochs": np.nan,
            "Time/Epochs": np.nan
        }
    )
    return data


def models_summary(custom, bench):
    models_data = pd.concat([bench, custom], ignore_index=True)

    # models_data["Distance (L-BFGS)"] = models_data["Solution"].apply(
    #     lambda x: np.linalg.norm(x - bench.loc[0]["Solution"]))

    models_data["Sol norm"] = models_data["Solution"].apply(
        lambda x: np.linalg.norm(x))

    return models_data.drop(columns={"Solution", "Fun/Epochs", "Time/Epochs"})


def plot_loss_time(ax, data, scalexy):
    df = data.copy()
    df.loc[:, "labels"] = df["Solver"] + \
        "(" + df["Alpha0"].astype(str) + ")"

    end = data["Fun/Epochs"][0].shape[0]
    indices = np.arange(0, end, 10)

    R = data.shape[0]
    for i in range(R//2):
        ax.plot(df["Time/Epochs"][i][indices], df["Fun/Epochs"][i][indices], linestyle="dashed")

    for i in range(R//2, R):
        ax.plot(df["Time/Epochs"][i][indices], df["Fun/Epochs"][i][indices], linestyle="solid")

    ax.set_xscale(scalexy[0])
    ax.set_yscale(scalexy[1])

    ax.grid(True, which="both", axis="both")
    ax.legend(df["labels"], fontsize="x-small")


def plot_loss_epochs(ax, data, scalexy):
    df = data.copy()
    df.loc[:, "labels"] = df["Solver"] + \
        "(" + df["Alpha0"].astype(str) + ")"

    start = 1  # only in np.range
    end = data["Fun/Epochs"][0].shape[0]

    R = data.shape[0]  # number of rows
    for i in range(R//2):
        ax.plot(np.arange(start, end), df["Fun/Epochs"][i][:-1], linestyle="dashed")

    for i in range(R//2, R):
        ax.plot(np.arange(start, end), df["Fun/Epochs"][i][:-1], linestyle="solid")

    ax.set_xscale(scalexy[0])
    ax.set_yscale(scalexy[1])

    ax.grid(True, which="both", axis="both")
    ax.legend(df["labels"], fontsize="x-small")


def diagnostic(models, scalexy=("log", "log", "linear", "log")):

    # models is a list of list of LogisticRegression
    # list of length 6
    # [sgdf, sgdd, sgdm, armijo, mslc, mslr]

    models_choose = [models[0] + models[3], models[1] + models[3],
                     models[2] + models[4], models[2] + models[5]]

    scalexy_epochs, scalexy_runtime = scalexy[:2], scalexy[2:]

    fig, axs = plt.subplots(2, 4, layout="constrained", sharey="row", sharex="row",
                            figsize=(6.4*2, 4.8*1.5))

    for i, ax in enumerate(axs.flat):
        if i in (0,1,2,3):  # first row
            # 1) Train loss against epochs
            plot_loss_epochs(ax, optim_data(models_choose[i % 4]), scalexy_epochs)
            ax.set_xticks([1, 10, 100])
            ax.set_xticklabels([1, 10, 100])

        elif i in (4,5,6,7):  # second row
            # 2) Train loss against runtime
            plot_loss_time(ax, optim_data(models_choose[i % 4]), scalexy_runtime)

    xlabel1 = "Epochs"
    xlabel2 = "Time (seconds)"
    for i in range(4):
        axs[0, i].set_xlabel(xlabel1)
        axs[1, i].set_xlabel(xlabel2)

    ylabel1 = r"$f(w)$"
    axs[0, 0].set_ylabel(ylabel1)
    axs[1, 0].set_ylabel(ylabel1)


# def plot_accuracy(models, ticks):
#     solvers_dict = {}
#     solvers_dict["Train score"] = [model.accuracy_train for model in models]
#     solvers_dict["Test score"] = [model.accuracy_test for model in models]
#     x = np.arange(len(models))
#     bar_width = 0.35
#     multiplier = 0
#     fig, ax1 = plt.subplots(ncols=1, layout="constrained")
#     for score, vals in solvers_dict.items():
#         offset = bar_width * multiplier
#         rects = ax1.bar(x + offset, vals, bar_width, label=score)
#         ax1.bar_label(rects, fmt="%.3f", fontsize="small", padding=3)
#         multiplier += 1
#     ax1.set_ylabel("Accuracy")
#     ax1.set_xticks(x + bar_width / 2, ticks, rotation=90)
#     ax1.legend()
#     ax1.set_ylim([0, 1])
#     ax1.grid(True)
#     plt.show()


# def plot_runtime(models, ticks):
#     solvers_dict = {}
#     solvers_dict["Run-time"] = [model.runtime for model in models]
#     x = np.arange(len(models))
#     bar_width = 0.35
#     multiplier = 0
#     fig, ax1 = plt.subplots(ncols=1, layout="constrained")
#     for score, vals in solvers_dict.items():
#         offset = bar_width * multiplier
#         rects = ax1.bar(x + offset, vals, bar_width, label=score)
#         ax1.bar_label(rects, fmt="%.4f", padding=3)
#         multiplier += 1
#     ax1.set_ylabel("Run-time")
#     ax1.set_xticks(x, ticks, rotation=90)
#     # ax1.legend()
#     # ax1.set_ylim([0, 1])
#     ax1.grid(True)
#     plt.show()
