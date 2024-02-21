
#%% Packages
# import time
# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

from models import LogisticRegression
from ml_utils import optim_data, diagnostic, optim_bench, models_summary
# from solvers_utils import f_and_df, logistic, logistic_der

#%% Apple quality dataset

# already with constant column
X_train_apple = pd.read_csv("datasets/apple_quality/apple_X_train.csv").values
y_train_apple = pd.read_csv("datasets/apple_quality/apple_y_train.csv").values.reshape(-1)
X_test_apple = pd.read_csv("datasets/apple_quality/apple_X_test.csv").values
y_test_apple = pd.read_csv("datasets/apple_quality/apple_y_test.csv").values.reshape(-1)

#%% Class LogisticRegression

model0_1 = LogisticRegression(solver="L-BFGS", C=1).fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple)
model0_2 = LogisticRegression(solver="Newton-CG", C=1).fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple)
model0_3 = LogisticRegression(solver="CG", C=1).fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple)

bench_data = optim_bench([model0_1, model0_2, model0_3])

#%% SGD-Fixed
k = 100
M1 = 8

model1_1 = LogisticRegression(solver="SGD-Fixed", C=1, epochs=k, minibatch=M1)
model1_1.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.1)
model1_2 = LogisticRegression(solver="SGD-Fixed", C=1, epochs=k, minibatch=M1)
model1_2.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.01)
model1_3 = LogisticRegression(solver="SGD-Fixed", C=1, epochs=k, minibatch=M1)
model1_3.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.001)

# %% SGD-Decreasing

model2_1 = LogisticRegression(solver="SGD-Decreasing", C=1, epochs=k, minibatch=M1)
model2_1.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=1)
model2_2 = LogisticRegression(solver="SGD-Decreasing", C=1, epochs=k, minibatch=M1)
model2_2.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.5)
model2_3 = LogisticRegression(solver="SGD-Decreasing", C=1, epochs=k, minibatch=M1)
model2_3.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.1)

# %% SGDM

model3_1 = LogisticRegression(solver="SGDM", C=1, epochs=k, minibatch=M1)
model3_1.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.1, momentum=0.9)
model3_2 = LogisticRegression(solver="SGDM", C=1, epochs=k, minibatch=M1)
model3_2.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.01, momentum=0.9)
model3_3 = LogisticRegression(solver="SGDM", C=1, epochs=k, minibatch=M1)
model3_3.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.001, momentum=0.9)

# %%

models2_data = optim_data([model1_1, model1_2, model1_3,
                           model2_1, model2_2, model2_3,
                           model3_1, model3_2, model3_3])

# %% SGD-Armijo

model4_1 = LogisticRegression(solver="SGD-Armijo", C=1, epochs=k, minibatch=M1)
model4_1.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=1)
model4_2 = LogisticRegression(solver="SGD-Armijo", C=1, epochs=k, minibatch=M1)
model4_2.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.5)
model4_3 = LogisticRegression(solver="SGD-Armijo", C=1, epochs=k, minibatch=M1)
model4_3.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.1)

# %% MSL-SGDM-C

model5_1 = LogisticRegression(solver="MSL-SGDM-C", C=1, epochs=k, minibatch=M1)
model5_1.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=1, momentum=0.9)
model5_2 = LogisticRegression(solver="MSL-SGDM-C", C=1, epochs=k, minibatch=M1)
model5_2.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.5, momentum=0.9)
model5_3 = LogisticRegression(solver="MSL-SGDM-C", C=1, epochs=k, minibatch=M1)
model5_3.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.1, momentum=0.9)

# %% MSL-SGDM-R

model6_1 = LogisticRegression(solver="MSL-SGDM-R", C=1, epochs=k, minibatch=M1)
model6_1.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=1, momentum=0.9)
model6_2 = LogisticRegression(solver="MSL-SGDM-R", C=1, epochs=k, minibatch=M1)
model6_2.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.5, momentum=0.9)
model6_3 = LogisticRegression(solver="MSL-SGDM-R", C=1, epochs=k, minibatch=M1)
model6_3.fit(X_train_apple, y_train_apple, X_test_apple, y_test_apple,
    step_size=0.1, momentum=0.9)

# %% DataFrames

# models = [optim_data([model1_1, model1_2, model4_1, model4_2]),
#           optim_data([model2_1, model2_2, model4_1, model4_2]),
#           optim_data([model3_1, model3_2, model5_1, model5_2]),
#           optim_data([model3_1, model3_2, model6_1, model6_2])]
# fig, axs = plt.subplots(2, 2, sharey=True, layout="constrained", figsize=(6.4, 4.8))
# i = 0
# for ax in axs.flat:
#     plot_loss(ax, models[i])
#     i += 1

# models1 = optim_data([])

models_data = optim_data([model1_1, model1_2, model1_3, model4_1, model4_2, model4_3,
            model2_1, model2_2, model2_3, model4_1, model4_2, model4_3,
            model3_1, model3_2, model3_3, model5_1, model5_2, model5_3,
            model3_1, model3_2, model3_3, model6_1, model6_2, model6_3])

diagnostic(optim_data([model1_1, model1_2, model1_3, model4_1, model4_2, model4_3]),
           optim_data([model2_1, model2_2, model2_3, model4_1, model4_2, model4_3]),
           optim_data([model3_1, model3_2, model3_3, model5_1, model5_2, model5_3]),
           optim_data([model3_1, model3_2, model3_3, model6_1, model6_2, model6_3]),
           model0_1)

all_models = models_summary(models_data, bench_data)
