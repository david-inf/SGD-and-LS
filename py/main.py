
#%% Packages
# import time
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
# from myLogisticRegression import myLogRegr
# from solvers import(l_bfgs_b, sgd_fixed, sgd_decreasing, sgd_armijo,
#                     sgdm, msl_sgdm_c, msl_sgdm_r)
from ml_utils import set_accuracy, optim_data, diagnostic, optim_bench

from models import LogisticRegression

#%% Apple quality dataset

# load with constant column
X_train_apple = pd.read_csv("datasets/apple_quality/apple_X_train.csv").values
y_train_apple = pd.read_csv("datasets/apple_quality/apple_y_train.csv").values.reshape(-1)
X_test_apple = pd.read_csv("datasets/apple_quality/apple_X_test.csv").values
y_test_apple = pd.read_csv("datasets/apple_quality/apple_y_test.csv").values.reshape(-1)

# intercept initial guess already added
# w0 = np.array([-4, 3, -1, 1, 0, 2, 2.5, -1])
rng = np.random.default_rng(42)
w0 = (5 + 5) * rng.random(8) - 5

#%% Class LogisticRegression

model0_1 = LogisticRegression().fit(w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple)
model0_2 = LogisticRegression(solver="Newton-CG").fit(w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple)
model0_3 = LogisticRegression(solver="CG").fit(w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple)

#%%

model1_1 = LogisticRegression(solver="SGD-Fixed", C=1)
model1_1.fit(w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.1)

model2_1 = LogisticRegression(solver="SGD-Decreasing", C=1)
model2_1.fit(w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=1)

model3_1 = LogisticRegression(solver="SGDM", C=1)
model3_1.fit(w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.1, momentum=0.9)

model4_1 = LogisticRegression(solver="SGD-Armijo", C=1)
model4_1.fit(w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=1)

model5_1 = LogisticRegression(solver="MSL-SGDM-C", C=1)
model5_1.fit(w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.1, momentum=0.9)

model6_1 = LogisticRegression(solver="MSL-SGDM-R", C=1)
model6_1.fit(w0, X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.1, momentum=0.9)


# models1_data = optim_data([model1_1, model2_1, model3_1, model4_1, model5_1, model6_1])
# models1_bench = optim_bench([model0_1, model0_2, model0_3])

# diagnostic(models1_data)
