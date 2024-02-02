# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:33:32 2024

@author: Utente
"""

import numpy as np
import pandas as pd
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
