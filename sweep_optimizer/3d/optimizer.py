#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 14:52:23 2018
This is the driver file for the sweep optimizer.
@author: tghaddar
"""
from build_global_subset_boundaries import build_global_subset_boundaries
from build_adjacency_matrix import build_adjacency
from utilities import get_ijk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import build_3d_adjacency as b3a
import numpy as np
import sweep_solver
import networkx as nx
import warnings
from scipy.optimize import minimize
warnings.filterwarnings("ignore")

plt.close("all")

num_total_cells = 12000
#Time to solve a cell (ns)
solve_cell = 2.0
#Time to communicate cell boundary info (ns)
t_comm = 3.0 

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

def plot_subset_boundaries(global_3d_subset_boundaries,num_subsets):
  fig = plt.figure(1)
  ax = fig.gca(projection='3d')
  subset_centers = []
  layer_colors = ['b','r']
  layer = 0
  for i in range(0,num_subsets):
  
    subset_boundary = global_3d_subset_boundaries[i]
    xmin = subset_boundary[0]
    xmax = subset_boundary[1]
    ymin = subset_boundary[2]
    ymax = subset_boundary[3]
    zmin = subset_boundary[4]
    zmax = subset_boundary[5]
    if (zmax == 10.0):
      layer = 1
    else:
      layer = 0
  
    center_x = (xmin+xmax)/2
    center_y = (ymin+ymax)/2
    center_z = (zmin+zmax)/2
  
    subset_centers.append([center_x, center_y, center_z])
  
    x = [xmin, xmax, xmax, xmin, xmin,xmax,xmax,xmin,xmin,xmin,xmin,xmin,xmax,xmax,xmin,xmin]
    y = [ymin, ymin, ymax, ymax, ymin,ymin,ymin,ymin,ymin,ymax,ymax,ymin,ymin,ymax,ymax,ymin]
    z = [zmin, zmin, zmin, zmin, zmin,zmin,zmax,zmax,zmin,zmin,zmax,zmax,zmax,zmax,zmax,zmax]
  
    ax.plot(x,y,z,layer_colors[layer])
    
    x2 = [xmax,xmax]
    y2 = [ymax,ymax]
    z2 = [zmax,zmin]
    ax.plot(x2,y2,z2,layer_colors[layer])
  
  plt.savefig("subset_plot.pdf")

def time_solver(param):
  x_cuts,y_cuts,z_cuts = param
  
  #Adding global z boundaries.
  z_cuts = [global_z_min] + z_cuts + [global_z_max]
  
  #Adding x global boundaries
  for i in range(0,len(x_cuts)):
    #x_cuts for this plane
    p_x_cuts = x_cuts[i]
    p_x_cuts = [global_x_min] + p_x_cuts + [global_x_max]
    x_cuts[i] = p_x_cuts
  
  for j in range(0,len(y_cuts)):
    for jp in range(0,len(y_cuts[j])):
      p_y_cuts = y_cuts[j][jp]
      p_y_cuts = [global_y_min] + p_y_cuts + [global_y_max]
      y_cuts[j][jp] = p_y_cuts

  #Building the global subset boundaries.
  global_3d_subset_boundaries = b3a.build_3d_global_subset_boundaries(N_x,N_y,N_z,x_cuts,y_cuts,z_cuts)
  
  cell_dist = sweep_solver.get_subset_cell_dist(num_total_cells,global_3d_subset_boundaries)
  
  adjacency_matrix = b3a.build_adjacency_matrix(x_cuts,y_cuts,z_cuts,num_row,num_col,num_plane)
  
  graphs = b3a.build_graphs(adjacency_matrix,num_row,num_col,num_plane)
  
  #We need to acquire a cost distribution (cell solve time + comm time for each node in each graph)
  graphs = sweep_solver.add_edge_cost(graphs,num_total_cells,global_3d_subset_boundaries,cell_dist,solve_cell,t_comm,num_row,num_col,num_plane)
  
  #Solving for the amount of time.
  all_graph_time,time = sweep_solver.compute_solve_time(graphs,solve_cell,cell_dist,num_total_cells,global_3d_subset_boundaries,num_row,num_col,num_plane)
  
  #Removing the global bounds.
  z_cuts.pop(0)
  z_cuts.pop()
  
  for i in range(0,len(x_cuts)):
    #x_cuts for this plane
    p_x_cuts = x_cuts[i]
    p_x_cuts.pop(0)
    p_x_cuts.pop()
    x_cuts[i] = p_x_cuts
  
  for j in range(0,len(y_cuts)):
    for jp in range(0,len(y_cuts[j])):
      p_y_cuts = y_cuts[j][jp]
      p_y_cuts.pop(0)
      p_y_cuts.pop()
      y_cuts[j][jp] = p_y_cuts
      
  param = (x_cuts,y_cuts,z_cuts)
  return time

#Assuming the order of cutting is z,x,y.
z_cuts = [5]
#X_cuts per plane.
x_cuts = [[3],[5]]
#y_cuts per column per plane.
y_cuts = [[[3],[5]], [[4],[7]]]

param = (x_cuts,y_cuts,z_cuts)

time = time_solver(param)

