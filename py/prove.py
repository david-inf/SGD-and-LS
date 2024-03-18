# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 09:45:20 2024

@author: Utente
"""

import time
from joblib import Parallel, delayed

# Define a simple function to square a number
def square(x):
    return x * x

# Define a list of numbers
numbers = [1, 2, 3, 4, 5]

# Sequential computation
start_time = time.time()
squared_sequential = [square(x) for x in numbers]
sequential_time = time.time() - start_time
print("Squared (Sequential):", squared_sequential)
print("Time taken (Sequential):", sequential_time, "seconds")

# Parallel computation using joblib
num_cores = 3  # Number of CPU cores to use in parallel
start_time = time.time()
squared_parallel = Parallel(n_jobs=num_cores)(delayed(square)(x) for x in numbers)
parallel_time = time.time() - start_time
print("Squared (Parallel):", squared_parallel)
print("Time taken (Parallel):", parallel_time, "seconds")
