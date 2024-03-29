# -*- coding: utf-8 -*-

import time
import numpy as np
from itertools import product
from joblib import Parallel, delayed

from models import LogisticRegression


# %% Grid search

def fit_model(params, solver, C, dataset):
    """ fit the model with the specified parameters """

    model = LogisticRegression(solver, C)

    # this to avoid useless computations
    # if solver in ("SGD-Fixed", "SGD-Decreasing", "SGDM"):
    #     batch_size, alpha, beta, *rest = params

    #     model.fit(dataset, batch_size, alpha, beta)

    # elif solver in ("SGD-Armijo", "MSL-SGDM-R"):
    #     batch_size, alpha, beta, delta_a, *rest = params

    #     model.fit(dataset, batch_size, alpha, beta, damp_armijo=delta_a)

    # elif solver == "MSL-SGDM-C":
    batch_size, alpha, beta, delta_a, delta_m = params

    model.fit(dataset, batch_size, alpha, beta, damp_armijo=delta_a,
              damp_momentum=delta_m)

    # test accuracy and loss
    performance = np.array([model.metrics_test[0], -model.fun])

    return performance, model, params


def prepare_grid(solver, batches, alphas, betas, delta_a, delta_m):
    """ prepare the parameter grid for all solvers
        returns dict"""

    param_grid = {}
    param_names = ["batch", "alpha", "beta", "delta_a", "delta_m"]

    if solver in ("SGD-Fixed", "SGD-Decreasing", "SGD-Armijo"):
        betas = (0,)
        delta_m = (0,)

    if solver in ("SGD-Fixed", "SGD-Decreasing", "SGDM"):
        delta_a = (0,)

    if solver in ("SGDM", "MSL-SGDM-R"):
        delta_m = (0,)

    param_values = [batches, alphas, betas, delta_a, delta_m]
    param_grid = dict(zip(param_names, param_values))

    return param_grid


def grid_search(solver, C, dataset, batches, alphas=(1, 0.1, 0.01),
                betas=(0.9,), delta_a=(0.5,), delta_m=(0.5,),
                output=True, parallel=True):
    # for betas, delta_a and delta_m is set just one value because of the
    # solvers that don't use those parameters, there would be more
    # combinations that it needed

    # combinations of all the parameters regardless of the solver being used
    param_grid = prepare_grid(solver, batches, alphas, betas, delta_a, delta_m)

    best_performance = np.array([0, float("inf")])  # accuracy and loss
    best_params = None
    best_model = None

    start_time = time.time()

    params_combos = product(*param_grid.values())

    results = []

    if parallel:
        results = Parallel(n_jobs=4)(delayed(fit_model)(params, solver, C, dataset)
                                     for params in params_combos)

    else:
        for params in params_combos:
            res = fit_model(params, solver, C, dataset)
            results.append(res)

    for performance, model, params in results:
        if compare_performance(performance, best_performance):
            best_performance = performance
            best_params = params
            best_model = model

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


# def grid_search_par(solver, C, dataset, batches, alphas=(1, 0.1, 0.01),
#                 betas=(0.9,), delta_a=(0.5,), delta_m=(0.5,), output=True):

#     param_grid = prepare_grid(solver, batches, alphas, betas, delta_a, delta_m)

#     best_performance = np.array([0, float("inf")])  # accuracy and loss
#     best_params = None
#     best_model = None

#     start_time = time.time()

#     params_combos = product(*param_grid.values())

#     results = Parallel(n_jobs=4)(delayed(fit_model)(params, solver, C, dataset)
#                                  for params in params_combos)

#     for performance, model, params in results:
#         if compare_performance(performance, best_performance):
#             best_performance = performance
#             best_params = params
#             best_model = model

#     end_time = time.time()
#     elapsed_time = end_time - start_time

#     if output:
#         combos = 1
#         for val in param_grid.values():
#             combos *= len(val)

#         param_names = ["batch", "alpha", "beta", "delta_a", "delta_m"]
#         print(f"{dict(zip(param_names, best_params))}" +
#               f"\nGrid search run-time (seconds): {elapsed_time:.6f}" +
#               f"\nNumber of combinations analyzed: {combos}")
#         print("-----")
#         print(best_model)

#     return best_model, best_params


def compare_performance(perf, best_perf):
    result = False

    compare = perf > best_perf

    if compare[0]:
        result = True

    elif perf[0] == best_perf[0] and compare[1]:
        result = True

    return result

# %% Solvers plot

def run_bench(dataset, C):
    bench1 = LogisticRegression("L-BFGS-B", C=C).fit(dataset=dataset)
    bench2 = LogisticRegression("Newton-CG", C=C).fit(dataset=dataset)
    bench3 = LogisticRegression("CG", C=C).fit(dataset=dataset)

    return [bench1, bench2, bench3]


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

    start_time = time.time()

    results = Parallel(n_jobs=2)(delayed(fit_model)(params, solver, C, dataset)
                                 for params in param_grid)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Run-time (seconds): {elapsed_time:.6f}")

    solvers_output = []
    for _, model, _ in results:
        solvers_output.append(model)

    return solvers_output
