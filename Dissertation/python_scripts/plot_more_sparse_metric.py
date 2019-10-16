#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:21:16 2019

@author: tghaddar
"""

import numpy as np
import matplotlib.pyplot as plt



num_subsets = [2,3,4,5,6,7,8,9,10]
max_time_reg = np.genfromtxt("more_sparse_fs_reg")
max_time_lb = np.genfromtxt("more_sparse_fs_lb")
max_time_lbd = np.genfromtxt("more_sparse_fs_lbd")
max_time_opt = np.genfromtxt("more_sparse_fs_best")
plt.figure()
plt.xlabel(r'$\sqrt{\rm{Number\ of\ Subsets}}$')
plt.ylabel("f")
plt.plot(num_subsets,max_time_reg,'--o',label="Reg")
plt.plot(num_subsets,max_time_lb,'--o',label="LB")
plt.plot(num_subsets,max_time_lbd,'--o',label="LBD")
plt.plot(num_subsets,max_time_opt,'--o',label="Bin")
plt.legend(loc="best")
plt.savefig("../../figures/more_sparse_metric_comp.pdf")