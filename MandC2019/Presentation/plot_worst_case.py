# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 21:40:20 2019

@author: tghad
"""
import sys

#sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
sys.path.append(r'C:\Users\tghad\Documents\GitHub\TarekGhaddarGraduateWork\sweep_optimizer\3d')
from build_global_subset_boundaries import build_global_subset_boundaries
from sweep_solver import plot_subset_boundaries_2d
import matplotlib.pyplot as plt
import numpy as np


x_cuts_lbd = np.genfromtxt("x_cuts_5_worst.csv",delimiter=",")
y_cuts_lbd = np.genfromtxt("y_cuts_5_worst.csv",delimiter=",")

num_row = 5
num_col = 5
num_subsets = num_row*num_col
boundaries_lbd = build_global_subset_boundaries(num_col-1,num_row-1,x_cuts_lbd,y_cuts_lbd)

plot_subset_boundaries_2d(boundaries_lbd,num_subsets,"../../figures/boundaries_worst_pres.pdf")
