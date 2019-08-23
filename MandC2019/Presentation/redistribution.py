#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
"""
Created on Fri Aug 23 12:31:02 2019

@author: tghaddar
"""
plt.close("all")

cells = [76,14,4,6,4,4,6,4,6,4,14,12,6,9,6,6,9,6,9,6,6,4,6,4,4,6,4,6,4,6,9,6,9,6,6,9,6,9,6,4,6,4,6,4,4,6,4,6,4,4,6,4,6,4 ,4,6,4 ,6 ,4,6 ,9,6,9,6,6,9 ,6,9,6,4 ,6,4,6,4,4 ,6,4,6,4,6,9,6,9,6,6,9,6,12,14, 4,6 ,6,4 ,6 ,4,4 ,6 ,4 ,14,76]

cells_reshaped = np.reshape(cells,(10,10))
cell_sum = 0.0
cdf = np.empty(11)
cdf[0] = 0
for i in range(0,10):
  cell_sum += sum(cells_reshaped[i])
  cdf[i+1] = cell_sum

total_cells = cdf[10]
x = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
avg = [76*x for x in range(1,10)]
plt.figure()
plt.xlabel("X Cut Line Location (cm)")
plt.ylabel("CDF of Cells Per Column")
plt.plot(x,cdf,'-o')
for i in range(1,10):
  x_i = [0,1]
  y_i = [avg[i-1],avg[i-1]]
  plt.plot(x_i,y_i,'--r')

plt.savefig("../../figures/spiderweb_redistribute_after.pdf")

plt.figure()
plt.xlabel("X Cut Line Location (cm)")
plt.ylabel("CDF of Cells Per Column")
plt.plot(x,cdf,'-o')
plt.savefig("../../figures/spiderweb_redistribute_before.pdf")