# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 19:28:02 2024

@author: Utente
"""

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
