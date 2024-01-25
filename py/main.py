# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 21:20:04 2024

@author: Utente
"""

# import numpy as np
# np.linalg.norm()
# import scipy.linalg as la
# la.norm()
import pandas as pd
# from sklearn.model_selection import train_test_split


## Iris dataset
from sklearn.datasets import load_iris

iris = load_iris()

irisDF = pd.DataFrame(iris.data, columns=iris.feature_names)
irisDF["Iris type"] = iris.target
irisDF["Iris name"] = irisDF["Iris type"].apply(
    lambda x: "sentosa" if x == 0 else ("versicolor" if x == 1 else "virginica"))

X = irisDF.iloc[:, 0:4]
y = irisDF["Iris name"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


