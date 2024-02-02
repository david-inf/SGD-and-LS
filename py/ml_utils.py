# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:33:32 2024

@author: Utente
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from solvers_utils import sigmoid


def predict(X, w, thresh=0.5):
    y_proba = sigmoid(np.dot(X, w))
    y_proba[y_proba > thresh] = 1
    y_proba[y_proba <= thresh] = -1
    return y_proba


def get_accuracy(model, X, y_true):
    # model: OptimizeResult
    return accuracy_score(y_true, predict(X, model.x))


def optim_data(models):
    models_data = pd.DataFrame(
        {
            "Solver": [model.solver for model in models],
            "Minibatch": [model.minibatch_size for model in models],
            "Step-size": [model.step_size for model in models],
            "Momentum": [model.momentum for model in models],
            "Loss": [model.fun for model in models],
            "Grad norm": [model.grad for model in models],
            "Run-time": [model.runtime for model in models],
            "Termination": [model.message for model in models],
            "Train accuracy": [model.accuracy_train for model in models],
            "Test accuracy": [model.accuracy_test for model in models]
        }
    )
    return models_data


def plot_loss(models, labels, title=None):
    fig, ax1 = plt.subplots(ncols=1, layout="constrained")
    ax1.set_title(title)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Training loss")
    ax1.set_yscale("log")
    i = 0
    for model in models:
        ax1.plot(model.fun_per_it, label=labels[i])
        i += 1
    ax1.legend()
    plt.show()


def plot_accuracy(models, ticks):
    solvers_dict = {}
    solvers_dict["Train score"] = [model.accuracy_train for model in models]
    solvers_dict["Test score"] = [model.accuracy_test for model in models]
    x = np.arange(len(models))
    bar_width = 0.25
    multiplier = 0
    fig, ax1 = plt.subplots(ncols=1, layout="constrained")
    for score, vals in solvers_dict.items():
        offset = bar_width * multiplier
        rects = ax1.bar(x + offset, vals, bar_width, label=score)
        ax1.bar_label(rects, fmt="%.2f", padding=3)
        multiplier += 1
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(x + bar_width / 2, ticks, rotation=90)
    ax1.legend()
    ax1.set_ylim([0, 1])
    ax1.grid(True)
    plt.show()
