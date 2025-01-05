# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:04:38 2021

@author: htchen
"""

# If this script is not run under spyder IDE, comment the following two lines.
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd

# Function to set scatter plot limits
def scatter_pts_2d(x, y):
    xmax = np.max(x)
    xmin = np.min(x)
    xgap = (xmax - xmin) * 0.2
    xmin -= xgap
    xmax += xgap

    ymax = np.max(y)
    ymin = np.min(y)
    ygap = (ymax - ymin) * 0.2
    ymin -= ygap
    ymax += ygap 

    return xmin, xmax, ymin, ymax

# Load dataset
dataset = pd.read_csv('data/hw7.csv').to_numpy(dtype=np.float64)
x = dataset[:, 0]
y = dataset[:, 1]

# Initial parameters
w = np.array([-0.1607108,  2.0808538,  0.3277537, -1.5511576])
alpha = 0.05
max_iters = 500

# Gradient descent using analytic method
for _ in range(1, max_iters):
    # Compute predictions
    y_pred = w[0] + w[1] * np.sin(w[2] * x + w[3])
    # Compute gradients
    grad_w0 = -2 * np.sum(y - y_pred)
    grad_w1 = -2 * np.sum((y - y_pred) * np.sin(w[2] * x + w[3]))
    grad_w2 = -2 * np.sum((y - y_pred) * w[1] * x * np.cos(w[2] * x + w[3]))
    grad_w3 = -2 * np.sum((y - y_pred) * w[1] * np.cos(w[2] * x + w[3]))
    # Update weights
    w[0] -= alpha * grad_w0
    w[1] -= alpha * grad_w1
    w[2] -= alpha * grad_w2
    w[3] -= alpha * grad_w3

# Generate predicted values using analytic method
xmin, xmax, ymin, ymax = scatter_pts_2d(x, y)
xt = np.linspace(xmin, xmax, 100)
yt1 = w[0] + w[1] * np.sin(w[2] * xt + w[3])

# Reset parameters for numeric method
w = np.array([-0.1607108,  2.0808538,  0.3277537, -1.5511576])

# Gradient descent using numeric method
for _ in range(1, max_iters):
    epsilon = 1e-5  # Small value for numerical gradient calculation
    grad_w = np.zeros_like(w)
    for i in range(len(w)):
        w_temp = w.copy()
        w_temp[i] += epsilon
        y_pred_plus = w_temp[0] + w_temp[1] * np.sin(w_temp[2] * x + w_temp[3])
        cost_plus = np.sum((y - y_pred_plus) ** 2)

        w_temp[i] -= 2 * epsilon
        y_pred_minus = w_temp[0] + w_temp[1] * np.sin(w_temp[2] * x + w_temp[3])
        cost_minus = np.sum((y - y_pred_minus) ** 2)

        grad_w[i] = (cost_plus - cost_minus) / (2 * epsilon)
    w -= alpha * grad_w

# Generate predicted values using numeric method
xt = np.linspace(xmin, xmax, 100)
yt2 = w[0] + w[1] * np.sin(w[2] * xt + w[3])

# Plotting results
fig = plt.figure(dpi=288)
plt.scatter(x, y, color='k', edgecolor='w', linewidth=0.9, s=60, zorder=3)
plt.plot(xt, yt1, linewidth=4, c='b', zorder=0, label='Analytic method')
plt.plot(xt, yt2, linewidth=2, c='r', zorder=0, label='Numeric method')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.show()
