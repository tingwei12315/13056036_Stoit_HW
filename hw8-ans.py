# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:04:38 2021

@author: htchen
"""
# If this script is not run under spyder IDE, comment the following two lines.
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 讀取資料集
hw8_csv = pd.read_csv('data/hw8.csv')
hw8_dataset = hw8_csv.to_numpy(dtype=np.float64)

# 資料的特徵與標籤
X0 = hw8_dataset[:, 0:2]  # 特徵空間 (x1, x2)
y = hw8_dataset[:, 2]     # 分類標籤 (-1, 1)

# 初始化權重向量 w 和偏置 b
w = np.zeros(2)  # 權重
b = 0            # 偏置

# 學習率
alpha = 0.01

# 最大迭代次數
max_iters = 1000

# 執行感知機算法 (Perceptron Learning Algorithm)
for _ in range(max_iters):
    for i in range(len(X0)):
        if y[i] * (np.dot(w, X0[i]) + b) <= 0:  # 錯誤分類條件
            w += alpha * y[i] * X0[i]          # 更新權重
            b += alpha * y[i]                  # 更新偏置

# 定義決策邊界函數
def decision_boundary(x1):
    return -(w[0] * x1 + b) / w[1]

# 繪製分類結果與決策邊界
fig = plt.figure(dpi=288)

# 繪製分類數據點
plt.plot(X0[y == 1, 0], X0[y == 1, 1], 'r.', label='$\omega_1$')  # 類別 1
plt.plot(X0[y == -1, 0], X0[y == -1, 1], 'b.', label='$\omega_2$')  # 類別 2

# 繪製分類邊界
x1_min, x1_max = np.min(X0[:, 0]) - 1, np.max(X0[:, 0]) + 1
x2_min, x2_max = np.min(X0[:, 1]) - 1, np.max(X0[:, 1]) + 1
x1_vals = np.linspace(x1_min, x1_max, 100)
x2_vals = decision_boundary(x1_vals)

# 著色區域
xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 200), np.linspace(x2_min, x2_max, 200))
Z = np.sign(w[0] * xx + w[1] * yy + b)
plt.contourf(xx, yy, Z, levels=0, colors=['lightblue', 'lightcoral'], alpha=0.3)

# 繪製分類邊界線
plt.plot(x1_vals, x2_vals, 'k-', label='Decision Boundary')

# 標籤與範圍設置
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('equal')
plt.legend()
plt.show()
