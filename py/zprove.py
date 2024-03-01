# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 11:54:59 2024

@author: Utente
"""

import matplotlib.pyplot as plt

# Create a figure and a grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharey='row')

# Plot on each subplot
for i, ax in enumerate(axs.flat):
    ax.plot([1, 2, 3], [1, 2, 3] if i < 2 else [10, 20, 30])
    ax.set_title(f'Subplot {i+1}')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

