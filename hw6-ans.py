# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:04:38 2021

@author: htchen
"""

from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import math
import numpy as np
import numpy.linalg as la
import cv2
import matplotlib.pyplot as plt
import pandas as pd


# calculate the eigenvalues and eigenvectors of a squared matrix
# the eigenvalues are decreasing ordered
def myeig(A, symmetric=False):
    if symmetric:
        lambdas, V = np.linalg.eigh(A)
    else:
        lambdas, V = np.linalg.eig(A)
    # lambdas, V may contain complex value
    lambdas_real = np.real(lambdas)
    sorted_idx = lambdas_real.argsort()[::-1] 
    return lambdas[sorted_idx], V[:, sorted_idx]


# class 1
mean1 = np.array([0, 5])
sigma1 = np.array([[0.3, 0.2],[0.2, 1]])
N1 = 200
X1 = np.random.multivariate_normal(mean1, sigma1, N1)

# class 2
mean2 = np.array([3, 4])
sigma2 = np.array([[0.3, 0.2],[0.2, 1]])
N2 = 100
X2 = np.random.multivariate_normal(mean2, sigma2, N2)

# m1: mean of class 1
# m2: mean of class 2
m1 = np.mean(X1, axis=0, keepdims=True)
m2 = np.mean(X2, axis=0, keepdims=True)

# write your code here: calculate within-class scatter matrix (Sw)
Sw = np.cov(X1, rowvar=False) * (N1 - 1) + np.cov(X2, rowvar=False) * (N2 - 1)

# calculate between-class scatter matrix (Sb)
Sb = (m2 - m1).T @ (m2 - m1)

# perform eigendecomposition
lambdas, V = myeig(np.linalg.inv(Sw) @ Sb)

# choose the direction vector (Fisher's linear discriminant)
w = V[:, 0]

# project data onto the line
y1 = X1 @ w
y2 = X2 @ w

# prepare projected points for visualization
proj1 = np.outer(y1, w) + m1
proj2 = np.outer(y2, w) + m2

plt.figure(dpi=288)

# plot original data points
plt.plot(X1[:, 0], X1[:, 1], 'r.')
plt.plot(X2[:, 0], X2[:, 1], 'g.')

# plot projected points
plt.plot(proj1[:, 0], proj1[:, 1], 'r-', label='Class 1 projection')
plt.plot(proj2[:, 0], proj2[:, 1], 'g-', label='Class 2 projection')

# draw the discriminant line
line_x = np.linspace(-6, 6, 100)
line_y = w[1] / w[0] * (line_x - m1[0, 0]) + m1[0, 1]
plt.plot(line_x, line_y, 'b--', label='Fisher discriminant line')

plt.legend()
plt.axis('equal')
plt.show()
