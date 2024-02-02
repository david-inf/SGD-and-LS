# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:06:08 2024

@author: Utente
"""

# merge with ml_utils
import matplotlib.pyplot as plt

def plotDiagnostic(lrs, labels):
    # lrs: list of myLogRegr object
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(7, 2.5), layout="constrained")
    ax1.set_title("Training loss against epochs")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Training loss")
    ax1.set_yscale("log")
    ax2.set_title("Gradient norm against epochs")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Gradient norm")
    ax2.set_yscale("log")
    # ax2.set_ylim([0, 100])  # function goes out of range
    i = 0
    for lr in lrs: 
        ax1.plot(lr.obj_seq, label=labels[i])
        ax2.plot(lr.grad_seq, label=labels[i])
        i += 1
    ax1.legend()
    ax2.legend()
    plt.show()
    
def plot_loss(models, labels, title=None):
    fig, ax1 = plt.subplots(ncols=1, layout="constrained")
    ax1.set_title(title)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Training loss")
    ax1.set_yscale("log")
    i = 0
    for model in models:
        ax1.plot(model.fun_per_it, label=labels[i])
        i += 1
    ax1.legend()
    plt.show()
    
def plot_accuracy(models, labels, title=None):
    # grafico a barre dell'accuracy
    # sulla y l'accuracy
    # sulla x barre raggruppate per train e test accuracy, al variare di un iperparametro
    fig, ax1 = plt.subplots(ncols=1, layout="constrained")
    ax1.set_title(title)
    # ax1.set_xlabel()
    # ax1.set_ylabel("Accuracy")
    ax1.set_xticks(labels)
    ax1.set_ylim([0, 1])
    width = 0.25
    multiplier = 0
    for model in models:
        ax1.bar()
    ax1.legend(["Train score", "Test score"])
    plt.show()
    

# def printDiagnostic(model):
#     # TODO: add accuracy
#     print(f"{model.solver} accuracy: {model2_accuracy1:.6f}" +
#           f"\nSolution: {model.coef_}" + f"\nLoss: {model.obj_}" +
#           f"\nGradient: {model.grad_}" +
#           "\nSolver message: " + model.solver_message)
