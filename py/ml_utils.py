# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:33:32 2024

@author: Utente
"""

import numpy as np
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
