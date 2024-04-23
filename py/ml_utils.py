# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

# %% DataFrames

def optim_data(models):
    # models: list of LogisticRegression

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
            "Fun/Epochs": [model.opt_result.fun_per_epoch for model in models],
            "Time/Epochs": [model.opt_result.time_per_epoch for model in models]
        }
    )

    return models_data


def optim_bench(models):
    # models: list of LogisticRegression

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


# %% Classification metrics

def confusion_matrix(y_true, y_pred):
    true_pos = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    false_neg = np.sum(np.logical_and(y_pred == -1, y_true == 1))

    false_pos = np.sum(np.logical_and(y_pred == 1, y_true == -1))
    true_neg = np.sum(np.logical_and(y_pred == -1, y_true == -1))

    return np.array([[true_pos, false_neg],
                     [false_pos, true_neg]])

def my_accuracy(y_true, y_pred):
    res = y_pred == y_true

    return np.sum(res) / res.size


def my_recall(y_true, y_pred):
    true_pos = confusion_matrix(y_true, y_pred)[0, 0]
    false_neg = confusion_matrix(y_true, y_pred)[0, 1]

    return true_pos / (true_pos + false_neg)


def my_specificity(y_true, y_pred):
    true_neg = confusion_matrix(y_true, y_pred)[1, 1]
    false_pos = confusion_matrix(y_true, y_pred)[1, 0]

    return true_neg / (true_neg + false_pos)


def my_bal_accuracy(y_true, y_pred):

    return 0.5 * (my_recall(y_true, y_pred) + my_specificity(y_true, y_pred))


def my_f1(y_true, y_pred):
    true_pos = confusion_matrix(y_true, y_pred)[0, 0]
    false_pos = confusion_matrix(y_true, y_pred)[1, 0]
    false_neg = confusion_matrix(y_true, y_pred)[0, 1]

    return 2. * true_pos / (2. * true_pos + false_pos + false_neg)


def metrics_list(y_true, y_pred):
    """ Returns a list with some metrics for classification """

    return [my_accuracy(y_true, y_pred),
            my_bal_accuracy(y_true, y_pred),
            my_f1(y_true, y_pred)]

# %% Plotting

# def plot_loss_time(ax, models, scalexy, multiple=True):
#     end = models[0].opt_result.time_per_epoch.shape[0]
#     indices = np.arange(0, end, 3)

#     models_deep = copy.deepcopy(models)
#     for model in models_deep:
#         seq = model.opt_result.time_per_epoch
#         for i in range(10):
#             if seq[i] <= 1e-3:
#                 seq[i] = 1e-3

#     for i in range(R):
#         time_seq = models_deep[i].opt_result.time_per_epoch[indices]
#         fun_seq = models_deep[i].opt_result.fun_per_epoch[indices]
#         ax.plot(time_seq, fun_seq, linestyle=lines[i])


# def plot_loss_epochs(ax, models, scalexy, multiple=True):
#     start = 1  # only in np.range
#     end = models[0].opt_result.fun_per_epoch.shape[0]

#     for i in range(R):
#         fun_seq = models[i].opt_result.fun_per_epoch
#         ax.plot(np.arange(start, end + 1), fun_seq, linestyle=lines[i])


def plot_loss(ax, models, scalexy, multiple=True, time_or_epoch=True):
    # models: list of LogisticRegression

    labels = []
    for model in models:
        res = model.opt_result
        labels.append(model.solver + f"({res.step_size:.2f}, {res.minibatch_size})")

    R = len(models)
    if multiple:
        lines = ["dashed"] * (R//2) + ["solid"] * (R - R//2)
    else:
        lines = ["solid"] * R

    # ---------------- #
    if time_or_epoch:  # f(w) against epochs
        start = 1  # due to logarithmic scale issues
        end = models[0].opt_result.fun_per_epoch.shape[0]

        for i in range(R):
            fun_seq = models[i].opt_result.fun_per_epoch
            ax.plot(np.arange(start, end + 1), fun_seq, linestyle=lines[i])

    else:  # f(w) against time per epoch
        end = models[0].opt_result.time_per_epoch.shape[0]
        indices = np.arange(0, end, 3)  # time every 3 epochs

        models_deep = copy.deepcopy(models)
        for model in models_deep:
            seq = model.opt_result.time_per_epoch
            for i in range(10):
                if seq[i] <= 1e-3:
                    seq[i] = 1e-3

        for i in range(R):
            time_seq = models_deep[i].opt_result.time_per_epoch[indices]
            fun_seq = models_deep[i].opt_result.fun_per_epoch[indices]
            ax.plot(time_seq, fun_seq, linestyle=lines[i])
    # ---------------- #

    ax.set_xscale(scalexy[0])
    ax.set_yscale(scalexy[1])

    ax.grid(True, which="both", axis="both")
    ax.legend(labels, fontsize="xx-small")


def diagnostic(models, scalexy=("log", "log", "log", "log")):
    # models: list of list of LogisticRegression
    # list of length 6
    # [sgdf, sgdd, sgdm, armijo, mslc, mslr]

    models_choose = [models[0] + models[3], models[1] + models[3],
                     models[2] + models[4], models[2] + models[5]]

    scalexy_epochs, scalexy_runtime = scalexy[:2], scalexy[2:]

    fig, axs = plt.subplots(2, 4, layout="constrained", sharey="row", sharex="row",
                            figsize=(6.4*2, 4.8*1.5))

    for i, ax in enumerate(axs.flat):
        if i < 4:  # first row
            # 1) f(w) against epochs
            # plot_loss_epochs(ax, optim_data(models_choose[i % 4]), scalexy_epochs)
            # plot_loss_epochs(ax, models_choose[i % 4], scalexy_epochs)
            plot_loss(ax, models_choose[i % 4], scalexy_epochs, time_or_epoch=True)
            ax.set_xticks([1, 10, 100])
            ax.set_xticklabels([0, 10, 100])

        else:  # second row
            # 2) f(w) against runtime per epoch
            # plot_loss_time(ax, optim_data(models_choose[i % 4]), scalexy_runtime)
            # plot_loss_time(ax, models_choose[i % 4], scalexy_runtime)
            plot_loss(ax, models_choose[i % 4], scalexy_runtime, time_or_epoch=False)
            ax.set_xticks([0.001, 0.01, 0.1, 1])
            ax.set_xticklabels([0, 0.01, 0.1, 1])

    xlabel1 = "Epochs"
    xlabel2 = "Run-time (seconds)"
    for i in range(4):
        axs[0, i].set_xlabel(xlabel1)
        axs[1, i].set_xlabel(xlabel2)

    ylabel1 = r"$f(w)$"
    axs[0, 0].set_ylabel(ylabel1)
    axs[1, 0].set_ylabel(ylabel1)


def single_diagnostic(models, scalexy=("log", "log", "log", "log")):
    # models: list of LogisticRegression

    scalexy_epochs, scalexy_runtime = scalexy[:2], scalexy[2:]

    fig, axs = plt.subplots(1, 2, layout="constrained", sharey="row")

    for i, ax in enumerate(axs.flat):
        if i == 0:  # first
            # 1) f(w) against epochs
            # plot_loss_epochs(ax, models, scalexy_epochs, False)
            plot_loss(ax, models, scalexy_epochs, False, True)
            ax.set_xticks([1, 10, 100])
            ax.set_xticklabels([0, 10, 100])

        elif i == 1:  # second
            # 2) f(w) against runtime per epoch
            # plot_loss_time(ax, models, scalexy_runtime, False)
            plot_loss(ax, models, scalexy_runtime, False, False)
            ax.set_xticks([0.001, 0.01, 0.1, 1])
            ax.set_xticklabels([0, 0.01, 0.1, 1])

    xlabel1 = "Epochs"
    xlabel2 = "Run-time (seconds)"
    axs[0].set_xlabel(xlabel1)
    axs[1].set_xlabel(xlabel2)

    ylabel1 = r"$f(w)$"
    axs[0].set_ylabel(ylabel1)
    axs[1].set_ylabel(ylabel1)

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
