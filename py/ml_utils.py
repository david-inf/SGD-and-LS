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


def set_accuracy(model, X_train, y_train, X_test, y_test):
    # model: OptimizeResult
    model.accuracy_train = accuracy_score(y_train, predict(X_train, model.x))
    model.accuracy_test = accuracy_score(y_test, predict(X_test, model.x))


def optim_data(models):
    models_data = pd.DataFrame(
        {
            "Solver": [model.solver for model in models],
            "Minibatch": [model.minibatch_size for model in models],
            "Step-size": [model.step_size for model in models],
            "Momentum": [model.momentum for model in models],
            "Solution": [model.x for model in models],
            "Loss": [model.fun for model in models],
            "Grad norm": [model.grad for model in models],
            "Run-time": [model.runtime for model in models],
            "Termination": [model.message for model in models],
            "Train accuracy": [model.accuracy_train for model in models],
            "Test accuracy": [model.accuracy_test for model in models]
        }
    )
    return models_data


def plot_loss(models, labels, title=None, start=5, end=200):
    fig, ax1 = plt.subplots(ncols=1, layout="constrained")
    i = 0
    for model in models:
        ax1.plot(np.arange(start, 200), model.fun_per_it[start:end], label=labels[i])
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


def diagnostic(models, labels, start_loss=5, end_loss=200):
    # plot_loss(models, labels, start=start_loss)
    # plot_runtime(models, labels)
    # plot_accuracy(models, labels)
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, layout="constrained")
    
    
    
    

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








