# -*- coding: utf-8 -*-

import numpy as np
# from scipy.sparse import csr_matrix, hstack

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, MaxAbsScaler

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error

from scipy.sparse import lil_matrix, hstack

from joblib import Memory

cachedir = "./tmp/mycache"
memory = Memory(cachedir, verbose=0)
# memory.clear(warn=False)  # keep them cached


# %% Utils

def dataset_distrib(y_train):
    values, counts = np.unique(y_train, return_counts=True)
    samples = np.sum(counts)

    return dict(zip(values, counts / samples))


def data_info(dataset):
    train_sklearn_log(dataset[0], dataset[1], dataset[2], dataset[3])


def train_sklearn_log(X_train, y_train, X_test, y_test):
    print(f"X_train = {X_train.shape}, y_train = {y_train.shape}")
    print(f"X_test = {X_test.shape}, y_test = {y_test.shape}")
    print(f"Train distribution: {dataset_distrib(y_train)}")
    print(f"Test distribution: {dataset_distrib(y_test)}")

    # model = LogisticRegression().fit(X_train, y_train)
    # train_score = accuracy_score(y_train, model.predict(X_train))
    # test_score = accuracy_score(y_test, model.predict(X_test))
    # bal_train_score = balanced_accuracy_score(y_train, model.predict(X_train))
    # bal_test_score = balanced_accuracy_score(y_test, model.predict(X_test))

    # print(f"sklearn train score: {train_score:.6f}")
    # print(f"sklearn test score: {test_score:.6f}")
    # print(f"sklearn balanced train score: {bal_train_score:.6f}")
    # print(f"sklearn balanced test score: {bal_test_score:.6f}")
    # weights = np.insert(model.coef_, 0, model.intercept_)
    # print(f"sklearn sol norm: {np.linalg.norm(weights)}")


def add_intercept(X):
    ones = lil_matrix(np.ones((X.shape[0], 1)))
    X_prep = hstack([ones, X.tolil()], format="csr")

    return X_prep


# TODO: add scaling function?

""" dataset_X = load_X() """

# TODO: use joblib memoization
# la posso usare quando memorizzo il dataset in dataset_X così lo butta sul disco
# però chiamando il load_X soltanto una volta servirebbe a poco memoizzare
# a meno che invece di salvare i dati nelle variabili, passo direttamente la funzione
# perché proprio quello che faccio visto a fit passo dataset_X = load_X()


# %% w1a

@memory.cache
def load_w1a(disp=False):  # ok
    path_train = "datasets/LIBSVM/w1a.txt"
    X_train, y_train = load_svmlight_file(path_train)

    path_test = "datasets/LIBSVM/w1a.t"
    X_test, y_test = load_svmlight_file(path_test)

    # add constant column
    X_train_prep = add_intercept(X_train)
    X_test_prep = add_intercept(X_test)

    if disp:
        train_sklearn_log(X_train_prep, y_train, X_test_prep, y_test)

    return X_train_prep, y_train, X_test_prep, y_test


# @memory.cache
def load_diabetes():  # ok
    path = "datasets/LIBSVM/diabetes_scale.txt"
    X, y = load_svmlight_file(path)

    # add constant column
    # ones = lil_matrix(np.ones((y.size, 1)))
    # X_prep = hstack([ones, X.tolil()], format="csr")
    X_prep = add_intercept(X)

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_prep, y, test_size=0.2, random_state=42)

    train_sklearn_log(X_train, y_train, X_test, y_test)

    return X_train, y_train, X_test, y_test


# %% w3a

@memory.cache
def load_w3a(disp=False):  # ok
    path_train = "datasets/LIBSVM/w3a.txt"
    X_train, y_train = load_svmlight_file(path_train)

    path_test = "datasets/LIBSVM/w3a.t"
    X_test, y_test = load_svmlight_file(path_test)

    # add constant column
    X_train_prep = add_intercept(X_train)
    X_test_prep = add_intercept(X_test)

    if disp:
        train_sklearn_log(X_train_prep, y_train, X_test_prep, y_test)

    return X_train_prep, y_train, X_test_prep, y_test


def load_breast_cancer():  # ok
    path = "datasets/LIBSVM/breast-cancer_scale.txt"
    X, y = load_svmlight_file(path)

    # add constant column
    X_prep = add_intercept(X)

    # enconde response variable in {-1,1}
    encoder = LabelEncoder()
    y_prep = 2 * encoder.fit_transform(y) - 1

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_prep, y_prep, test_size=0.2, random_state=42)

    train_sklearn_log(X_train, y_train, X_test, y_test)

    return X_train, y_train, X_test, y_test


# %% phishing

@memory.cache
def load_phishing(disp=False):
    path = "datasets/LIBSVM/phishing.txt"
    X, y = load_svmlight_file(path)

    # add constant column
    X_prep = add_intercept(X)

    # encode respponse variable in {-1,1}
    encoder = LabelEncoder()
    y_prep = 2 * encoder.fit_transform(y) - 1

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_prep, y_prep, test_size=0.2, random_state=42)

    if disp:
        train_sklearn_log(X_train, y_train, X_test, y_test)

    return X_train, y_train, X_test, y_test


def load_svmguide1():  # ok
    path_train = "datasets/LIBSVM/svmguide1.txt"
    X_train, y_train = load_svmlight_file(path_train)

    path_test = "datasets/LIBSVM/svmguide1.t"
    X_test, y_test = load_svmlight_file(path_test)

    # scale dataset in [-1,1]
    # scaler = MinMaxScaler((-1, 1)).fit(X_train)
    # X_train_scaled = scaler.transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    # scale dataset
    scaler = MaxAbsScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # add constant column
    X_train_prep = add_intercept(X_train_scaled)
    X_test_prep = add_intercept(X_test_scaled)

    # encode respponse variable in {-1,1}
    encoder = LabelEncoder()
    y_train_prep = 2 * encoder.fit_transform(y_train) - 1
    y_test_prep = 2 * encoder.fit_transform(y_test) - 1

    train_sklearn_log(X_train_prep, y_train_prep, X_test_prep, y_test_prep)

    return X_train_prep, y_train_prep, X_test_prep, y_test_prep


# %% a2a

@memory.cache
def load_a2a(disp=False):  # ok
    path_train = "datasets/LIBSVM/a2a.txt"
    X_train, y_train = load_svmlight_file(path_train)

    path_test = "datasets/LIBSVM/a2a.t"
    X_test, y_test = load_svmlight_file(path_test)

    # add constant column
    X_train_prep = add_intercept(X_train)
    X_test_prep = add_intercept(X_test)[:, :120]

    if disp:
        train_sklearn_log(X_train_prep, y_train, X_test_prep, y_test)

    return X_train_prep, y_train, X_test_prep, y_test


def load_australian():  # ok
    path = "datasets/LIBSVM/australian_scale.txt"
    X, y = load_svmlight_file(path)

    # add constant column
    X_prep = add_intercept(X)

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_prep, y, test_size=0.2, random_state=42)

    train_sklearn_log(X_train, y_train, X_test, y_test)

    return X_train, y_train, X_test, y_test


# %% Mushrooms

@memory.cache
def load_mushrooms(disp=False):  # ok
    path = "datasets/LIBSVM/mushrooms.txt"
    X, y = load_svmlight_file(path)

    # add constant column
    X_prep = add_intercept(X)

    # encode respponse variable in {-1,1}
    encoder = LabelEncoder()
    y_prep = 2 * encoder.fit_transform(y) - 1

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_prep, y_prep, test_size=0.2, random_state=42)

    if disp:
        train_sklearn_log(X_train, y_train, X_test, y_test)

    return X_train, y_train, X_test, y_test


# %% German

# @memory.cache
def load_german(disp=False):  # ok
    path = "datasets/LIBSVM/german.numer_scale.txt"
    X, y = load_svmlight_file(path)

    # add constant column
    X_prep = add_intercept(X)

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_prep, y, test_size=0.2, random_state=42)

    if disp:
        train_sklearn_log(X_train, y_train, X_test, y_test)

    return X_train, y_train, X_test, y_test


# %% More datasets

def load_w6a():  # ok
    path_train = "datasets/LIBSVM/w6a.txt"
    X_train, y_train = load_svmlight_file(path_train)

    path_test= "datasets/LIBSVM/w6a.t"
    X_test, y_test= load_svmlight_file(path_test)

    # add constant column
    X_train_prep = add_intercept(X_train)
    X_test_prep = add_intercept(X_test)

    train_sklearn_log(X_train_prep, y_train, X_test_prep, y_test)

    return X_train_prep, y_train, X_test_prep, y_test


def load_a3a():
    path_train = "datasets/LIBSVM/a3a.txt"
    X_train, y_train = load_svmlight_file(path_train)

    path_test = "datasets/LIBSVM/a3a.t"
    X_test, y_test = load_svmlight_file(path_test)

    # add constant column
    # X_train_prep = np.hstack((np.ones((X_train_arr.shape[0],1)), X_train_arr))
    # X_test_prep = np.hstack((np.ones((X_test_arr.shape[0],1)), X_test_arr))

    # print(f"X_train = {X_train_prep.shape}, y_train = {y_train.shape}")
    # print(f"X_test = {X_test_prep.shape}, y_test = {y_test.shape}")
    # print(f"Class distribution: {dataset_distrib(y_train)}")
    # return X_train_prep, y_train, X_test_prep, y_test
    return X_train, y_train, X_test, y_test


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
@memory.cache
def load_mg(disp=False):
    path = "datasets/LIBSVM/mg_scale.txt"
    X, y = load_svmlight_file(path)

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # add constant column
    X_train_prep = add_intercept(X_train)
    X_test_prep = add_intercept(X_test)

    if disp:
        train_sklearn_log(X_train_prep, y_train, X_test_prep, y_test)
    
    return X_train_prep, y_train, X_test_prep, y_test
