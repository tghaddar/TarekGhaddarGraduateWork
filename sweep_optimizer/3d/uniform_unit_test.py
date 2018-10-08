#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 11:14:51 2018
#Uniform unit test. Tests a lot of things for uniform layouts.
@author: tghaddar
"""

from build_3d_adjacency import build_3d_global_subset_boundaries
from sweep_solver import plot_subset_boundaries
import warnings
warnings.filterwarnings("ignore")

#Number of cuts in the x direction.
N_x = 1
#Number of cuts in the y direction.
N_y = 1
#Number of cuts in the z direction.
N_z = 1
#Total number of subsets
num_subsets = (N_x+1)*(N_y+1)*(N_z+1)
num_subsets_2d = (N_x+1)*(N_y+1)

#Global bounds.
global_x_min = 0.0
global_x_max = 10.0

global_y_min = 0.0
global_y_max = 10.0

global_z_min = 0.0
global_z_max = 10.0

num_row = N_y + 1
num_col = N_x + 1
num_plane = N_z + 1

#Assuming the order of cutting is z,x,y.
z_cuts = [global_z_min,5,global_z_max]
#X_cuts per plane.
x_cuts = [[global_x_min,5,global_x_max],[global_x_min,5,global_x_max]]
#y_cuts per column per plane.
y_cuts = [[[global_y_min,5,global_y_max],[global_y_min,5,global_y_max]], [[global_y_min,5,global_y_max],[global_y_min,5,global_y_max]]]

global_subset_boundaries = build_3d_global_subset_boundaries(N_x,N_y,N_z,x_cuts,y_cuts,z_cuts)
fig = plot_subset_boundaries(global_subset_boundaries,num_subsets)

