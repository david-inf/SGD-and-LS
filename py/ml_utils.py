# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
# import matplotlib
# matplotlib.use("pgf")
import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score
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


# def set_accuracy(model, X_train, y_train, X_test, y_test):
#     # model: OptimizeResult
#     model.accuracy_train = accuracy_score(y_train, predict(X_train, model.x))
#     model.accuracy_test = accuracy_score(y_test, predict(X_test, model.x))


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
    models_data = pd.concat([bench, custom], ignore_index=True)
    models_data["Distance (L-BFGS)"] = models_data["Solution"].apply(
        lambda x: np.linalg.norm(x - bench.loc[0]["Solution"]))
    return models_data.drop(columns={"Solution", "Loss/Epochs"})


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


def plot_loss(ax, data):
    df = data.copy()
    df.loc[:, "labels"] = df["Solver"] + "(" + df["Step-size"].astype(str) + ")"
    start = 0
    end = data["Loss/Epochs"][0].shape[0]
    R = data.shape[0]
    for i in range(R//2):
        ax.plot(np.arange(start, end), df["Loss/Epochs"][i], linestyle="dashed")
    for i in range(R//2, R):
        ax.plot(np.arange(start, end), df["Loss/Epochs"][i], linestyle="solid")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Train loss")
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="both")
    ax.set_ylim(top=10 ** 0)
    ax.legend(df["labels"], fontsize="x-small")


def diagnostic(data1, data2, data3, data4):
    models = [data1, data2, data3, data4]
    fig, axs = plt.subplots(2, 2, layout="constrained", sharey=True, sharex=True,
                            figsize=(6.4, 4.8))
    i = 0
    for ax in axs.flat:
        plot_loss(ax, models[i])
        i += 1
