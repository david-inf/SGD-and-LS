# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 09:45:20 2024

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt

# Original vector
vector = np.linspace(0.01, 3, 100)  # Assuming you want 100 points between 0.01 and 3

# Plot the original data on linear scale
plt.figure(figsize=(8, 6))
# plt.plot(vector, label='Original Data (Linear Scale)')

# Use logarithmic scale for the y-axis (vertical axis)
# plt.yscale("log")
plt.xscale('log')

### posso moltiplicare per 100

# Plot the data on a logarithmic scale
plt.plot(vector, label='Data on Logarithmic Scale')

plt.xlim(0.01, 3)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Using Logarithmic Scale on Data')
plt.legend()
plt.grid(True)
plt.show()
