# -*- coding: utf-8 -*-

import time
import numpy as np
from itertools import product
from joblib import Parallel, delayed

from models import LogisticRegression

# %% cross-validation

def cross_val(model, data, k=5, solver_options={}):
    # data: (X_train, y_train, X_test, y_test)
    # k: number of folds

    # dataset size
    X_train, y_train, _, _ = data
    N = X_train.shape[0]  # int
    # get indices for splitting
    folds_idx = np.array_split(np.arange(N), k)  # list of numpy.ndarray of np.int32

    scores = np.empty(k)
    for i in range(k):
        this_folds_idx = folds_idx.copy()

        validate_idx = this_folds_idx.pop(i)
        fold_X_validate = X_train[validate_idx]
        fold_y_validate = y_train[validate_idx]

        folds_X_train = X_train[np.concatenate(this_folds_idx)]
        folds_y_train = y_train[np.concatenate(this_folds_idx)]

        fold_data = (folds_X_train, folds_y_train, fold_X_validate, fold_y_validate)

        scores[i] = model.fit(fold_data, **solver_options).fun  # not working

    return np.mean(scores)

# %% Grid search

def fit_model(params, solver, C, dataset, plot=False):
    """ fit the model with the specified parameters """

    # get all the model parameters
    batch_size, alpha, beta, delta_a, delta_m = params

    model = LogisticRegression(solver, C)

    if plot:
        model.fit(dataset, batch_size, alpha, beta, stop=0, max_epochs=200,
                  damp_armijo=delta_a, damp_momentum=delta_m)

    else:
        model.fit(dataset, batch_size, alpha, beta, damp_armijo=delta_a,
                  damp_momentum=delta_m)

    # test accuracy and loss
    # performance = model.metrics_test[0]
    performance = np.array([model.metrics_test[0], model.fun])

    return performance, model, params

# TODO
def fit_model_cv(params, solver, C, dataset, plot=False):
    """ cross-validate the model with the specified parameters """

    # get all the model parameters
    batch_size, alpha, beta, delta_a, delta_m = params

    model = LogisticRegression(solver, C)

    if plot:
        model.fit(dataset, batch_size, alpha, beta, stop=0, max_epochs=200,
                  damp_armijo=delta_a, damp_momentum=delta_m)

    else:
        model.fit(dataset, batch_size, alpha, beta, damp_armijo=delta_a,
                  damp_momentum=delta_m)

    # test accuracy and loss
    # performance = model.metrics_test[0]
    performance = np.array([model.metrics_test[0], model.fun])

    return performance, model, params


def prepare_grid(solver, batches, alphas, betas, delta_a, delta_m):
    """ prepare the parameter grid for all solvers
        returns dict"""

    param_grid = {}
    param_names = ["batch", "alpha", "beta", "delta_a", "delta_m"]

    if solver in ("SGD-Fixed", "SGD-Decreasing", "SGD-Armijo", "Adam"):
        betas = (0,)
        delta_m = (0,)

    if solver in ("SGD-Fixed", "SGD-Decreasing", "SGDM", "Adam"):
        delta_a = (0,)

    if solver in ("SGDM", "MSL-SGDM-R"):
        delta_m = (0,)

    param_values = [batches, alphas, betas, delta_a, delta_m]
    param_grid = dict(zip(param_names, param_values))

    return param_grid


# dumb grid search without cross-validation
# TODO: use k-fold cross-validation for tuning
def grid_search(solver, C, dataset, batches, alphas=(1, 0.1, 0.01),
                betas=(0.9,), delta_a=(0.5,), delta_m=(0.5,),
                output=True, do_parallel=True, plot=False, n_jobs=7):
    # for betas, delta_a and delta_m is set just one value because of the
    # solvers that don't use those parameters, there would be more
    # combinations that it needed

    # combinations of all the parameters regardless of the solver being used
    param_grid = prepare_grid(solver, batches, alphas, betas, delta_a, delta_m)

    # best_performance = np.array([0, float("inf")])  # accuracy and loss
    best_model = None   # store best model
    best_params = None  # store best model parameters

    start_time = time.time()

    # all possible combinations
    params_combos = product(*param_grid.values())

    results = []  # list of tuples

    if do_parallel:
        # I'm reusing the same pool through Parallel context manager
        with Parallel(n_jobs=n_jobs, backend="loky") as parallel:
            results = parallel(
                delayed(fit_model)(params, solver, C, dataset, plot)
                for params in params_combos)
            # results = performance, model, params

    else:
        for params in params_combos:
            res = fit_model(params, solver, C, dataset)
            results.append(res)

    ### comparison
    # sort elements by test (validation) accuracy and loss
    results.sort(key=lambda x: (x[0][0], -x[0][1]))
    best_model = results[-1][1]
    best_params = results[-1][2]

    end_time = time.time()
    elapsed_time = end_time - start_time

    if output:
        combos = 1
        for val in param_grid.values():
            combos *= len(val)

        param_names = ["batch", "alpha", "beta", "delta_a", "delta_m"]
        print(f"{dict(zip(param_names, best_params))}" +
              f"\nGrid search run-time (seconds): {elapsed_time:.6f}" +
              f"\nNumber of combinations analyzed: {combos}")
        print("-----")
        print(best_model)

    return best_model, best_params


# not scalable, make more general
# def compare_performance(perf, best_perf):
#     result = False

#     if perf[0] > best_perf[0]:
#         result = True

#     elif perf[0] == best_perf[0] and perf[1] < best_perf[1]:
#         result = True

#     return result


# %% Solvers plot

def run_bench(dataset, C):
    bench1 = LogisticRegression("L-BFGS-B", C=C).fit(dataset=dataset)
    bench2 = LogisticRegression("Newton-CG", C=C).fit(dataset=dataset)
    bench3 = LogisticRegression("CG", C=C).fit(dataset=dataset)

    return [bench1, bench2, bench3]


def run_solvers(solver, C, dataset, batches, alphas=(1,0.1,0.01),
                betas=(0.9,), delta_a=(0.5,), delta_m=(0.5,),
                do_parallel=False, n_jobs=2, **kwargs):

    # for each learning rate value, chooses the other best hyper-params

    start_time = time.time()

    # def run_solvers_fit():

    ### grid search fine-tuning
    solvers_output = []

    # if do_parallel:
    #     # I'm reusing the same pool through Parallel context manager
    #     with Parallel(n_jobs=n_jobs, backend="loky") as parallel:
    #         results = parallel(
    #             delayed(fit_model)(params, solver, C, dataset, plot)
    #             for params in params_combos)
    #         # results = performance, model, params

    # else:
    for i in range(3):
        model, _ = grid_search(solver, C, dataset, batches, (alphas[i],),
                               betas, delta_a, delta_m, output=False,
                               n_jobs=n_jobs, plot=True)
        solvers_output.append(model)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Run-time (seconds): {elapsed_time:.6f}")

    return solvers_output
