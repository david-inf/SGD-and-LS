# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
# import matplotlib
# matplotlib.use("pgf")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from solvers_utils import sigmoid

# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
#     "pgf.preamble": "\n".join([
#          r"\usepackage{lmodern}",
#          r"\usepackage[utf8]{inputenc}",
#          r"\usepackage[T1]{fontenc}",
#     ]),
# })


def predict(X, w, thresh=0.5):
    y_proba = sigmoid(np.dot(X, w))
    y_proba[y_proba > thresh] = 1
    y_proba[y_proba <= thresh] = -1
    return y_proba


def set_accuracy(model, X_train, y_train, X_test, y_test):
    # model: OptimizeResult
    model.accuracy_train = accuracy_score(y_train, predict(X_train, model.x))
    model.accuracy_test = accuracy_score(y_test, predict(X_test, model.x))


def optim_data(models):
    # models: LogisticRegression
    models_data = pd.DataFrame(
        {
            "Solver": [model.solver for model in models],
            "C": [model.C for model in models],
            "Minibatch": [model.minibatch for model in models],
            "Step-size": [model.opt_result.step_size for model in models],
            "Momentum": [model.opt_result.momentum for model in models],
            "Solution": [np.round(model.coef_, 4) for model in models],
            "Loss": [model.loss for model in models],
            "Grad norm": [model.grad for model in models],
            "Run-time": [model.opt_result.runtime for model in models],
            "Iterations": [model.opt_result.nit for model in models],
            # "Termination": [model.message for model in models],
            "Train accuracy": [model.accuracy_train for model in models],
            "Test accuracy": [model.accuracy_test for model in models],
            "Loss/Epochs": [model.loss_seq for model in models]
        }
    )
    return models_data


def optim_bench(models):
    data = pd.DataFrame(
        {
            "Solver": [model.solver for model in models],
            "C": [model.C for model in models],
            "Minibatch": np.nan,
            "Step-size": np.nan,
            "Momentum": np.nan,
            "Solution": [np.round(model.coef_, 4) for model in models],
            "Loss": [model.loss for model in models],
            "Grad norm": [model.grad for model in models],
            "Run-time": np.nan,
            "Iterations": [model.opt_result.nit for model in models],
            # "Termination": model.message,
            "Train accuracy": [model.accuracy_train for model in models],
            "Test accuracy": [model.accuracy_test for model in models],
            "Loss/Epochs": np.nan
        }
    )
    return data


def models_summary(custom, bench):
    models_data = pd.concat([bench.drop(columns={"Loss/Epochs"}),
                             custom.drop(columns={"Loss/Epochs"})],
                            ignore_index=True)
    models_data["Distance (L-BFGS)"] = models_data["Solution"].apply(
        lambda x: np.linalg.norm(x - bench.loc[0]["Solution"]))
    return models_data.drop(columns="Solution")


def plot_loss(models, labels, title=None, start=0, end=200):
    fig, ax1 = plt.subplots(ncols=1, layout="constrained")
    i = 0
    for model in models:
        ax1.plot(np.arange(start, end),
                 model.fun_per_it[start:end], label=labels[i])
        # ax1.plot(model.fun_per_it[start:], label=labels[i])
        i += 1
    ax1.set_title(title)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Training loss")
    ax1.set_yscale("log")
    # ax1.set_ylim([0, 1])
    # ax1.set_xticklabels(np.arange(25, 201, 25))
    ax1.legend()
    plt.show()


def plot_accuracy(models, ticks):
    solvers_dict = {}
    solvers_dict["Train score"] = [model.accuracy_train for model in models]
    solvers_dict["Test score"] = [model.accuracy_test for model in models]
    x = np.arange(len(models))
    bar_width = 0.35
    multiplier = 0
    fig, ax1 = plt.subplots(ncols=1, layout="constrained")
    for score, vals in solvers_dict.items():
        offset = bar_width * multiplier
        rects = ax1.bar(x + offset, vals, bar_width, label=score)
        ax1.bar_label(rects, fmt="%.3f", fontsize="small", padding=3)
        multiplier += 1
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(x + bar_width / 2, ticks, rotation=90)
    ax1.legend()
    ax1.set_ylim([0, 1])
    ax1.grid(True)
    plt.show()


def plot_runtime(models, ticks):
    solvers_dict = {}
    solvers_dict["Run-time"] = [model.runtime for model in models]
    x = np.arange(len(models))
    bar_width = 0.35
    multiplier = 0
    fig, ax1 = plt.subplots(ncols=1, layout="constrained")
    for score, vals in solvers_dict.items():
        offset = bar_width * multiplier
        rects = ax1.bar(x + offset, vals, bar_width, label=score)
        ax1.bar_label(rects, fmt="%.4f", padding=3)
        multiplier += 1
    ax1.set_ylabel("Run-time")
    ax1.set_xticks(x, ticks, rotation=90)
    # ax1.legend()
    # ax1.set_ylim([0, 1])
    ax1.grid(True)
    plt.show()


def test_plots(data):
    fig, axs = plt.subplots(3, 1, layout="constrained", figsize=(10, 12))
    data["labels"] = data["Solver"] + "(" + data["Step-size"].astype(str) + ")"
    # 1) Training loss
    start = 0
    end = data["Loss/Epochs"][0].shape[0]
    for loss in data["Loss/Epochs"]:
        axs[0].plot(np.arange(start, end), loss[start:end])
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Training loss")
    axs[0].set_yscale("log")
    axs[0].legend(data["labels"])
    axs[0].grid(True)
    # 2) Run-time
    x = np.arange(data.shape[0])
    bar_width = 0.35
    rects1 = axs[1].bar(data["labels"], data["Run-time"], bar_width)
    axs[1].grid(True)
    axs[1].set_ylabel("Run-time (seconds)")
    axs[1].bar_label(rects1, fmt="%.4f", padding=3)
    axs[1].set_xticks(x, data["labels"], rotation=90)
    # 3) Accuracy
    accuracy_dict = {}
    accuracy_dict["Train score"] = data["Train accuracy"]
    accuracy_dict["Test score"] = data["Test accuracy"]
    multiplier = 0
    for score, vals in accuracy_dict.items():
        offset = bar_width * multiplier
        rects2 = axs[2].bar(x + offset, vals, bar_width, label=score)
        axs[2].bar_label(rects2, fmt="%.3f", fontsize="small", padding=3)
        multiplier += 1
    axs[2].grid(True)
    axs[2].set_ylabel("Accuracy")
    axs[2].set_xticks(x + bar_width / 2, data["labels"], rotation=90)
    axs[2].set_ylim([0, 1])
    axs[2].legend()


def diagnostic(data):
    fig, ax = plt.subplots(layout="constrained", figsize=(6.4, 4.8))
    df = data
    df["labels"] = df["Solver"] + "(" + df["Step-size"].astype(str) + ")"
    # 1) Training loss
    start = 1
    end = df["Loss/Epochs"][0].shape[0]
    for loss in df["Loss/Epochs"]:
        ax.plot(np.arange(start, end), loss[start:end])
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Training loss")
    ax.set_yscale("log")
    ax.legend(df["labels"])
    ax.grid(True, which="both", axis="both")
    ax.set_ylim(top=10 ** 0)  # bug
    # ax.set_clip_on(True)
    # plt.show()

# test_plots(models1_data)


# def diagnostic_plots(models, start_loss=5, end_loss=101):
#     # models: list of LogisticRegression
#     fig, axs = plt.subplots(3, 1, layout="constrained",
#                                         figsize=(10, 10))
#     # 1) Training loss
#     for model in models:
#         axs[0].plot(np.arange(start_loss, end_loss),
#                  model.loss_seq[start_loss:end_loss], label=model.solver)
#     # for model in models[3:]:
#     #     axs[0].plot(np.arange(start_loss, end_loss),
#     #              model.loss_seq[start_loss:end_loss], label=model.solver,
#     #              linestyle="dashed")
#     axs[0].set_xlabel("Epochs")
#     axs[0].set_ylabel("Training loss")
#     axs[0].set_yscale("log")
#     axs[0].set_ylim([0.1, 5])
#     axs[0].grid(True)
#     axs[0].legend()
    # 2) Runtime
    # x = np.arange(len(models))
    # bar_width = 0.35
    # multiplier1 = 0
    # for model in models:
    #     offset1 = bar_width * multiplier1
    #     rects1 = axs[1].bar(x + offset1, model.runtime, bar_width, label=model.solver)
    #     axs[1].bar_label(rects1, fmt="%.4f", padding=3)
    #     multiplier1 += 1
    # axs[1].set_ylabel("Run-time")
    # axs[1].set_xticks(x, [model.solver + f"({model.opt_result.step_size})" for model in models],
    #                   rotation=90)
    # axs[1].grid(True)
    # 3) Accuracy
    # multiplier2 = 0
    # rects2_1 = axs[2].bar(x + offset, model.accuracy_train, bar_width, label="Train score")
    # axs[2].bar_label(rects2_1, fmt="%.3f", fontsize="small", padding=3)
    # rects2_2 = axs[2].bar(x + offset, model.accuracy_test, bar_width, label="Test score")
    # axs[2].bar_label(rects2_2, fmt="%.3f", fontsize="small", padding=3)
    # multiplier2 += 1
    # axs[2].set_ylabel("Accuracy")
    # axs[2].set_xticks(x + bar_width / 2,
    #         [model.solver + f"({model.opt_result.step_size})" for model in models], rotation=90)
    # axs[2].legend()
    # axs[2].set_ylim([0, 1])
    # axs[2].grid(True)


# def train_comparison(optimizer, *args, **kwargs):
#     # optimizer: callable
#     # the first kw should the hyperparameter on which one wuold test the sensitivity
#     # and the values should be given with a list
#     # returns a list of models
#     models = []
#     param = kwargs.popitem()  # pop the last element
#     for val in param[1]:
#         model = optimizer(param[0]=val, X=args[0], y=args[1], kwargs)
#         set_accuracy(model, args[0], args[1], args[2], args[3])
#         models.append(model)
#     return models

# train_comparison(minibatch_gd_fixed, , w0=w0, alpha[1, 0.1])
