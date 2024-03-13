# -*- coding: utf-8 -*-

import numpy as np
# from scipy.sparse import csr_matrix, hstack

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error



# Utils

def dataset_distrib(y_train):
    values, counts = np.unique(y_train, return_counts=True)
    samples = np.sum(counts)

    return dict(zip(values, counts / samples))


""" dataset_X = load_X() """


# %% Diabetes

def load_diabetes():  # ok
    path = "datasets/LIBSVM/diabetes_scale.txt"
    X, y = load_svmlight_file(path)
    # transform to array from CSR sparse matrix
    X_arr = X.toarray()

    # add constant column
    X_prep = np.hstack((np.ones((X_arr.shape[0],1)), X_arr))

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.2, random_state=42)

    print(f"X_train = {X_train.shape}, y_train = {y_train.shape}")
    print(f"X_test = {X_test.shape}, y_test = {y_test.shape}")
    print(f"Class distribution: {dataset_distrib(y_train)}")

    model = LogisticRegression().fit(X_train, y_train)
    train_score = accuracy_score(y_train, model.predict(X_train))
    test_score = accuracy_score(y_test, model.predict(X_test))

    print(f"sklearn train score: {train_score:.6f}")
    print(f"sklearn test score: {test_score:.6f}")
    weights = np.insert(model.coef_, 0, model.intercept_)
    print(f"sklearn sol norm: {np.linalg.norm(weights)}")

    return X_train, y_train, X_test, y_test


# %% Breast cancer

def load_breast_cancer():  # ok
    path = "datasets/LIBSVM/breast-cancer_scale.txt"
    X, y = load_svmlight_file(path)
    # transform to array from CSR sparse matrix
    X_arr = X.toarray()

    # add constant column
    X_prep = np.hstack((np.ones((X_arr.shape[0],1)), X_arr))

    # enconde response variable in {-1,1}
    encoder = LabelEncoder()
    y_prep = 2 * encoder.fit_transform(y) - 1

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_prep, y_prep, test_size=0.2, random_state=42)

    print(f"X_train = {X_train.shape}, y_train = {y_train.shape}")
    print(f"X_test = {X_test.shape}, y_test = {y_test.shape}")
    print(f"Class distribution: {dataset_distrib(y_train)}")

    model = LogisticRegression().fit(X_train, y_train)
    train_score = accuracy_score(y_train, model.predict(X_train))
    test_score = accuracy_score(y_test, model.predict(X_test))

    print(f"sklearn train score: {train_score:.6f}")
    print(f"sklearn test score: {test_score:.6f}")
    weights = np.insert(model.coef_, 0, model.intercept_)
    print(f"sklearn sol norm: {np.linalg.norm(weights)}")

    return X_train, y_train, X_test, y_test


# %% svmguide1

def load_svmguide1():  # ok
    path_train = "datasets/LIBSVM/svmguide1.txt"
    X_train, y_train = load_svmlight_file(path_train)
    # transform to array from CSR sparse matrix
    X_train_arr = X_train.toarray()

    path_test = "datasets/LIBSVM/svmguide1.t"
    X_test, y_test = load_svmlight_file(path_test)
    # transform to array from CSR sparse matrix
    X_test_arr = X_test.toarray()

    # scale dataset in [-1,1]
    scaler = MinMaxScaler((-1, 1)).fit(X_train_arr)
    X_train_scaled = scaler.transform(X_train_arr)
    X_test_scaled = scaler.transform(X_test_arr)

    # add constant column
    X_train_prep = np.hstack((np.ones((X_train_scaled.shape[0],1)), X_train_scaled))
    X_test_prep = np.hstack((np.ones((X_test_scaled.shape[0],1)), X_test_scaled))

    # encode respponse variable in {-1,1}
    encoder = LabelEncoder()
    y_train_prep = 2 * encoder.fit_transform(y_train) - 1
    y_test_prep = 2 * encoder.fit_transform(y_test) - 1

    print(f"X_train = {X_train_prep.shape}, y_train = {y_train_prep.shape}")
    print(f"X_test = {X_test_prep.shape}, y_test = {y_test_prep.shape}")
    print(f"Class distribution: {dataset_distrib(y_train_prep)}")

    model = LogisticRegression().fit(X_train_prep, y_train_prep)
    train_score = accuracy_score(y_train_prep, model.predict(X_train_prep))
    test_score = accuracy_score(y_test_prep, model.predict(X_test_prep))

    print(f"sklearn train score: {train_score:.6f}")
    print(f"sklearn test score: {test_score:.6f}")
    weights = np.insert(model.coef_, 0, model.intercept_)
    print(f"sklearn sol norm: {np.linalg.norm(weights)}")

    return X_train_prep, y_train_prep, X_test_prep, y_test_prep


# %% Australian

def load_australian():  # ok
    path = "datasets/LIBSVM/australian_scale.txt"
    X, y = load_svmlight_file(path)
    # transform to array from CSR sparse matrix
    X_arr = X.toarray()

    X_prep = np.hstack((np.ones((X_arr.shape[0],1)), X_arr))

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.2, random_state=42)

    print(f"X_train = {X_train.shape}, y_train = {y_train.shape}")
    print(f"X_test = {X_test.shape}, y_test = {y_test.shape}")
    print(f"Class distribution: {dataset_distrib(y_train)}")

    model = LogisticRegression().fit(X_train, y_train)
    train_score = accuracy_score(y_train, model.predict(X_train))
    test_score = accuracy_score(y_test, model.predict(X_test))

    print(f"sklearn train score: {train_score:.6f}")
    print(f"sklearn test score: {test_score:.6f}")
    weights = np.insert(model.coef_, 0, model.intercept_)
    print(f"sklearn sol norm: {np.linalg.norm(weights)}")

    return X_train, y_train, X_test, y_test


# %% Mushrooms

def load_mushrooms():  # ok
    path = "datasets/LIBSVM/mushrooms.txt"
    X, y = load_svmlight_file(path)
    # transform to array from CSR sparse matrix
    X_arr = X.toarray()

    # add constant column
    X_prep = np.hstack((np.ones((X_arr.shape[0],1)), X_arr))

    # encode respponse variable in {-1,1}
    encoder = LabelEncoder()
    y_prep = 2 * encoder.fit_transform(y) - 1

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_prep, y_prep, test_size=0.2, random_state=42)

    print(f"X_train = {X_train.shape}, y_train = {y_train.shape}")
    print(f"X_test = {X_test.shape}, y_test = {y_test.shape}")
    print(f"Class distribution: {dataset_distrib(y_train)}")

    model = LogisticRegression().fit(X_train, y_train)
    train_score = accuracy_score(y_train, model.predict(X_train))
    test_score = accuracy_score(y_test, model.predict(X_test))

    print(f"sklearn train score: {train_score:.6f}")
    print(f"sklearn test score: {test_score:.6f}")
    weights = np.insert(model.coef_, 0, model.intercept_)
    print(f"sklearn sol norm: {np.linalg.norm(weights)}")

    return X_train, y_train, X_test, y_test


# %% German

def load_german():  # ok
    path = "datasets/LIBSVM/german.numer_scale.txt"
    X, y = load_svmlight_file(path)
    # transform to array from CSR sparse matrix
    X_arr = X.toarray()

    # add constant column
    X_prep = np.hstack((np.ones((X_arr.shape[0],1)), X_arr))

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.2, random_state=42)

    print(f"X_train = {X_train.shape}, y_train = {y_train.shape}")
    print(f"X_test = {X_test.shape}, y_test = {y_test.shape}")
    print(f"Class distribution: {dataset_distrib(y_train)}")

    model = LogisticRegression().fit(X_train, y_train)
    train_score = accuracy_score(y_train, model.predict(X_train))
    test_score = accuracy_score(y_test, model.predict(X_test))

    print(f"sklearn train score: {train_score:.6f}")
    print(f"sklearn test score: {test_score:.6f}")
    weights = np.insert(model.coef_, 0, model.intercept_)
    print(f"sklearn sol norm: {np.linalg.norm(weights)}")

    return X_train, y_train, X_test, y_test


# %% More datasets


def load_w2a():
    path = "datasets/LIBSVM/w2a.txt"
    X, y = load_svmlight_file(path)
    # transform to array from CSR sparse matrix
    X_arr = X.toarray()
    return X_arr, y


def load_w4a():  # ok
    path_train = "datasets/LIBSVM/w4a.txt"
    X_train, y_train = load_svmlight_file(path_train)
    # transform to array from CSR sparse matrix
    # X_arr = X.toarray()

    path_test= "datasets/LIBSVM/w4a.t"
    X_test, y_test= load_svmlight_file(path_test)
    # transform to array from CSR sparse matrix
    # X_arr = X.toarray()

    print(f"X_train = {X_train.shape}, y_train = {y_train.shape}")
    print(f"X_test = {X_test.shape}, y_test = {y_test.shape}")
    print(f"Class distribution: {dataset_distrib(y_train)}")

    model = LogisticRegression().fit(X_train, y_train)
    train_score = accuracy_score(y_train, model.predict(X_train))
    test_score = accuracy_score(y_test, model.predict(X_test))

    print(f"sklearn train score: {train_score:.6f}")
    print(f"sklearn test score: {test_score:.6f}")
    weights = np.insert(model.coef_, 0, model.intercept_)
    print(f"sklearn sol norm: {np.linalg.norm(weights)}")

    return X_train.toarray(), y_train, X_test.toarray(), y_test


def load_w5a():  # ok
    path_train = "datasets/LIBSVM/w5a.txt"
    X_train, y_train = load_svmlight_file(path_train)
    # transform to array from CSR sparse matrix
    # X_arr = X.toarray()

    path_test= "datasets/LIBSVM/w5a.t"
    X_test, y_test= load_svmlight_file(path_test)
    # transform to array from CSR sparse matrix
    # X_arr = X.toarray()

    print(f"X_train = {X_train.shape}, y_train = {y_train.shape}")
    print(f"X_test = {X_test.shape}, y_test = {y_test.shape}")
    print(f"Class distribution: {dataset_distrib(y_train)}")

    model = LogisticRegression().fit(X_train, y_train)
    train_score = accuracy_score(y_train, model.predict(X_train))
    test_score = accuracy_score(y_test, model.predict(X_test))

    print(f"sklearn train score: {train_score:.6f}")
    print(f"sklearn test score: {test_score:.6f}")
    weights = np.insert(model.coef_, 0, model.intercept_)
    print(f"sklearn sol norm: {np.linalg.norm(weights)}")

    return X_train.toarray(), y_train, X_test.toarray(), y_test


def load_a3a():  # ok
    path_train = "datasets/LIBSVM/a3a.txt"
    X_train, y_train = load_svmlight_file(path_train)
    # transform to array from CSR sparse matrix
    # X_train_arr = X_train.toarray()

    path_test = "datasets/LIBSVM/a3a.t"
    X_test, y_test = load_svmlight_file(path_test)
    # transform to array from CSR sparse matrix
    # X_test_arr = X_test.toarray()

    # add constant column
    # X_train_prep = np.hstack((np.ones((X_train_arr.shape[0],1)), X_train_arr))
    # X_test_prep = np.hstack((np.ones((X_test_arr.shape[0],1)), X_test_arr))

    # print(f"X_train = {X_train_prep.shape}, y_train = {y_train.shape}")
    # print(f"X_test = {X_test_prep.shape}, y_test = {y_test.shape}")
    # print(f"Class distribution: {dataset_distrib(y_train)}")
    # return X_train_prep, y_train, X_test_prep, y_test
    return X_train.toarray(), y_train, X_test.toarray(), y_test


def load_a5a():  # ok
    path_train = "datasets/LIBSVM/a5a.txt"
    X_train, y_train = load_svmlight_file(path_train)
    # transform to array from CSR sparse matrix
    # X_train_arr = X_train.toarray()

    path_test = "datasets/LIBSVM/a5a.t"
    X_test, y_test = load_svmlight_file(path_test)
    # transform to array from CSR sparse matrix
    # X_test_arr = X_test.toarray()

    # add constant column
    # X_train_prep = np.hstack((np.ones((X_train_arr.shape[0],1)), X_train_arr))
    # X_test_prep = np.hstack((np.ones((X_test_arr.shape[0],1)), X_test_arr))

    # print(f"X_train = {X_train_prep.shape}, y_train = {y_train.shape}")
    # print(f"X_test = {X_test_prep.shape}, y_test = {y_test.shape}")
    # print(f"Class distribution: {dataset_distrib(y_train)}")
    # return X_train_prep, y_train, X_test_prep, y_test
    return X_train, y_train, X_test, y_test


def load_a4a():
    path_train = "datasets/LIBSVM/a4a.txt"
    X_train, y_train = load_svmlight_file(path_train)
    # transform to array from CSR sparse matrix
    # X_train_arr = X_train.toarray()

    path_test = "datasets/LIBSVM/a4a.t"
    X_test, y_test = load_svmlight_file(path_test)
    # transform to array from CSR sparse matrix
    # X_test_arr = X_test.toarray()

    # add constant column
    # X_train_prep = np.hstack((np.ones((X_train_arr.shape[0],1)), X_train_arr))
    # X_test_prep = np.hstack((np.ones((X_test_arr.shape[0],1)), X_test_arr))

    # print(f"X_train = {X_train_prep.shape}, y_train = {y_train.shape}")
    # print(f"X_test = {X_test_prep.shape}, y_test = {y_test.shape}")
    # print(f"Class distribution: {dataset_distrib(y_train)}")
    # return X_train_prep, y_train, X_test_prep, y_test
    return X_train, y_train, X_test, y_test


# Regression
def load_mg():
    path = "datasets/LIBSVM/mg_scale.txt"
    X, y = load_svmlight_file(path)
    # transform to array from CSR sparse matrix
    X_arr = X.toarray()

    # add constant column
    X_prep = np.hstack((np.ones((X_arr.shape[0],1)), X_arr))

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.2, random_state=42)

    print(f"X_train = {X_train.shape}, y_train = {y_train.shape}")
    print(f"X_test = {X_test.shape}, y_test = {y_test.shape}")

    model = LinearRegression().fit(X_train, y_train)
    mse = mean_squared_error(y_test, model.predict(X_test))

    print(f"sklearn MSE: {mse:.6f}")
    weights = np.insert(model.coef_, 0, model.intercept_)
    print(f"sklearn sol norm: {np.linalg.norm(weights)}")
    
    return X_train, y_train, X_test, y_test
