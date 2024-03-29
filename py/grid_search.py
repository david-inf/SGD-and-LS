# -*- coding: utf-8 -*-

import time
from itertools import product
from joblib import Parallel, delayed

from models import LogisticRegression


# %% Grid search

def grid_search(params, solver, C, dataset):
    """ fit the model with the specified parameters """

    model = LogisticRegression(solver, C)

    if solver in ("SGD-Fixed", "SGD-Decreasing", "SGDM"):
        batch_size, alpha, beta = params

        model.fit(dataset, batch_size, alpha, beta)

    elif solver == "SGD-Armijo":
        batch_size, alpha, beta, delta_a = params

        model.fit(dataset, batch_size, alpha, beta, damp_armijo=delta_a)

    elif solver == "MSL-SGDM-R":
        batch_size, alpha, beta, delta_a = params

        model.fit(dataset, batch_size, alpha, beta, damp_armijo=delta_a)

    elif solver == "MSL-SGDM-C":
        batch_size, alpha, beta, delta_a, delta_m = params

        model.fit(dataset, batch_size, alpha, beta, damp_armijo=delta_a, damp_momentum=delta_m)

    performance = model.fun

    return performance, model, params


def prepare_grid(solver, batches, alphas, betas, delta_a, delta_m):
    """ prepare the parameter grid for a certain solver
        returns dict"""

    param_grid = {}

    if solver in ("SGD-Fixed", "SGD-Decreasing", "SGD-Armijo"):
        betas = (0,)

    if solver in ("SGD-Fixed", "SGD-Decreasing", "SGDM"):
        param_grid = {"batch":batches, "alpha":alphas, "beta":betas}

    elif solver == "SGD-Armijo":
        param_grid = {"batch":batches, "alpha":alphas, "beta":betas,
                      "delta_a":delta_a}

    elif solver == "MSL-SGDM-R":
        param_grid = {"batch":batches, "alpha":alphas, "beta":betas,
                      "delta_a":delta_a}

    elif solver == "MSL-SGDM-C":
        param_grid = {"batch":batches, "alpha":alphas, "beta":betas,
                      "delta_a":delta_a, "delta_m":delta_m}

    return param_grid


def grid_search_seq(solver, C, dataset, batches, alphas=(1, 0.1, 0.01, 0.001),
                betas=(0.9,), delta_a=(0.5, 0.9), delta_m=(0.5,), output=True):

    param_grid = prepare_grid(solver, batches, alphas, betas, delta_a, delta_m)

    best_performance = float("inf")
    best_params = None
    best_model = None

    start_time = time.time()

    for params in product(*param_grid.values()):
        performance, model, params = grid_search(params, solver, C, dataset)

        if performance < best_performance:
            best_performance = performance
            best_params = params
            best_model = model

    end_time = time.time()
    elapsed_time = end_time - start_time

    if output:
        print(f"{params}" + f"\nGrid search run-time (seconds): {elapsed_time:.6f}")
        print("-----")
        print(best_model)

    return best_model, best_params


def grid_search_par(solver, C, dataset, batches, alphas=(1, 0.1, 0.01, 0.001),
                betas=(0.9,), delta_a=(0.5, 0.9), delta_m=(0.5,), output=True):

    param_grid = prepare_grid(solver, batches, alphas, betas, delta_a, delta_m)

    best_performance = float("inf")
    best_params = None
    best_model = None

    hyperparameter_combinations = product(*param_grid.values())

    start_time = time.time()

    results = Parallel(n_jobs=4)(delayed(grid_search)(params, solver, C, dataset)
                                 for params in hyperparameter_combinations)

    for perf, model, params in results:
        if perf < best_performance:
            best_performance = perf
            best_params = params
            best_model = model

    end_time = time.time()
    elapsed_time = end_time - start_time

    if output:
        print(f"{params}" + f"\nGrid search run-time (seconds): {elapsed_time:.6f}")
        print("-----")
        print(best_model)

    return best_model, best_params

# %% Solvers plot

def run_solvers(solver, C, dataset, batch_size, step_size=(1,0.1,0.01),
                momentum=(0.9,0.9,0.9), delta_a=0.5, gamma_a=0.001, delta_m=0.5,
                **kwargs):

    if solver in ("SGD-Fixed", "SGD-Decreasing", "SGD-Armijo"):
        momentum = (0, 0, 0)

    solvers_output = []

    start_time = time.time()

    model = None
    for i in range(3):
        model = LogisticRegression(solver, C=C)
        model.fit(dataset, batch_size, step_size[i], momentum[i], 0, 200,
                   delta_a, gamma_a, delta_m)
        solvers_output.append(model)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Run-time (seconds): {elapsed_time:.6f}")

    return solvers_output


def run_solvers_par(solver, C, dataset, batch_size, step_size=(1,0.1,0.01),
                momentum=(0.9,0.9,0.9), delta_a=0.5, gamma_a=0.001, delta_m=0.5,
                **kwargs):

    if solver in ("SGD-Fixed", "SGD-Decreasing", "SGD-Armijo"):
        momentum = (0, 0, 0)

    param_grid = []
    for i in range(3):
        param_grid.append((batch_size, step_size[i], momentum[i], delta_a, delta_m))

    def fit_model(params, solver, C, dataset):
        model = LogisticRegression(solver, C)
        batch_size, alpha, beta, delta_a, delta_m = params
        model.fit(dataset, batch_size, alpha, beta, damp_armijo=delta_a, damp_momentum=delta_m)
        return model

    start_time = time.time()

    results = Parallel(n_jobs=2)(delayed(fit_model)(params, solver, C, dataset)
                                 for params in param_grid)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Run-time (seconds): {elapsed_time:.6f}")

    solvers_output = []
    for model in results:
        solvers_output.append(model)

    return solvers_output
