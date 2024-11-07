# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:38:53 2024

@author: htchen
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# SVD: A = U * Sigma * V^T
# V: eigenvector matrix of A^T * A; U: eigenvector matrix of A * A^T 
def mysvd(A):
    lambdas, V = np.linalg.eigh(A.T @ A)
    lambdas, V = np.real(lambdas), np.real(V)
    lambdas = np.append(lambdas, 1e-12)
    rank = np.argwhere(lambdas < 1e-6).min()
    lambdas, V = lambdas[0:rank], V[:, 0:rank]
    U = A @ V / np.sqrt(lambdas)
    Sigma = np.diag(np.sqrt(lambdas))
    return U, Sigma, V

# Initialize parameters
pts = 50
x = np.linspace(-2, 2, pts)
y = np.zeros(x.shape)

# Create square wave signal
pts2 = pts // 2
y[0:pts2] = -1
y[pts2:] = 1

# Sort x values for plotting
argidx = np.argsort(x)
x = x[argidx]
y = y[argidx]

# Set Fourier series parameters
T0 = np.max(x) - np.min(x)
f0 = 1.0 / T0
omega0 = 2.0 * np.pi * f0
n = 5  # Number of Fourier components

# Step 1: Generate X matrix with cosine and sine terms
X = np.ones((pts, 2 * n + 1))  # Initialize X with the intercept term (column of ones)
for i in range(1, n + 1):
    X[:, i] = np.cos(i * omega0 * x)
    X[:, n + i] = np.sin(i * omega0 * x)

# Step 2: SVD of X
U, Sigma, VT = np.linalg.svd(X, full_matrices=False)

# Step 3: Calculate coefficients a using the pseudo-inverse
# Convert Sigma to a diagonal matrix for inversion
Sigma_inv = np.diag(1 / Sigma)  # Take reciprocal of diagonal elements
a = VT.T @ Sigma_inv @ U.T @ y

# Predict the square wave values using the Fourier series
y_bar = X @ a

# Plot the true and predicted values
plt.plot(x, y_bar, 'g-', label='Predicted values (Fourier Series)')
plt.plot(x, y, 'b-', label='True values (Square wave)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Square Wave Approximation Using Fourier Series')
plt.show()
