#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 10:41:34 2018

@author: tghaddar

Test space for 3D adjacency matrix building.
"""

from build_global_subset_boundaries import build_global_subset_boundaries
from build_adjacency_matrix import build_adjacency
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def get_ijk(ss_id,num_row,num_col,num_plane):
#  k = ss_id%num_plane
#  j = int((ss_id - k)/num_plane%num_row)
#  i = int(((ss_id-k)/num_plane - j)/num_row)
  k = int(ss_id/(num_row*num_col))
  if (ss_id >= num_row*num_col):
    ss_id -= (k)*num_row*num_col
  j = int(ss_id % num_row)
  i = int((ss_id - j)/num_row)
  
  
  return i,j,k

def get_all_ijk(num_subsets,num_row,num_col,num_plane):
  
  all_ijk = []
  for s in range(0,num_subsets):
    
    coords = get_ijk(s,num_row,num_col,num_plane)
    all_ijk.append(coords)
  
  return all_ijk

def build_3d_global_subset_boundaries(N_x,N_y,N_z,x_cuts,y_cuts,z_cuts):
  
  global_subset_boundaries = []
  num_subsets = (N_x+1)*(N_y+1)*(N_z+1)
  num_row = N_y + 1
  num_col = N_x + 1
  num_plane = N_z + 1
  
  for s in range(0,num_subsets):
    subset_boundary = []
    ss_id = s
    i,j,k = get_ijk(ss_id,num_row,num_col,num_plane)
    
    x_min = x_cuts[k][i]
    x_max = x_cuts[k][i+1]
    
    y_min = y_cuts[k][i][j]
    y_max = y_cuts[k][i][j+1]

    z_min = z_cuts[k]
    z_max = z_cuts[k+1]
    
    subset_boundary = [x_min,x_max,y_min,y_max,z_min,z_max]
    
    global_subset_boundaries.append(subset_boundary)
    
  return global_subset_boundaries
    

#Checking if two rectangles overlap.
def overlap(s_bounds,n_bounds):
  #The top left points of both the current subset and potential neighbor.
  s_topleft = [s_bounds[0],s_bounds[3]]
  n_topleft = [n_bounds[0],n_bounds[3]]
  #The bottom right points of both the current subset and potential neighbor.
  s_botright = [s_bounds[1],s_bounds[2]]
  n_botright = [n_bounds[1],n_bounds[2]]
  
  #If one rectangle is left of the other.
  if ( (s_topleft[0] > n_botright[0]) or (n_topleft[0] > s_botright[0]) ):
    return False
  
  #If one rectangle is above the other.
  if ( (s_topleft[1] < n_botright[1]) or (n_topleft[1] < s_botright[1]) ):
    return False
  
  return True

#We will be building this layer by layer. 

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
x_cuts = [[global_x_min,3,global_x_max],[global_x_min,5,global_x_max]]
#y_cuts per column per plane.
y_cuts = [[[global_y_min,3,global_y_max],[global_y_min,5,global_y_max]], [[global_y_min,4,global_y_max],[global_y_min,7,global_y_max]]]

#Building the global subset boundaries.
global_3d_subset_boundaries = build_3d_global_subset_boundaries(N_x,N_y,N_z,x_cuts,y_cuts,z_cuts)

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

all_2d_matrices = []
#Building an adjacency matrix for each layer.
adjacency_matrix_3d = np.zeros((num_subsets,num_subsets))
for z in range(0,num_plane):
  x_cuts_plane = x_cuts[z]
  y_cuts_plane = y_cuts[z]
  global_2d_subset_boundaries = build_global_subset_boundaries(N_x, N_y,x_cuts_plane,y_cuts_plane)
  adjacency_matrix = build_adjacency(global_2d_subset_boundaries,N_x,N_y,y_cuts_plane)
  
  adjacency_matrix_3d[z*num_subsets_2d:(z+1)*num_subsets_2d,z*num_subsets_2d:(z+1)*num_subsets_2d] = adjacency_matrix
  all_2d_matrices.append(adjacency_matrix_3d)

#Time to add in the neighbors in 3D. We'll loop over the subsets in each layer and add in neighbors that are in the layer above and below.
all_ijk = get_all_ijk(num_subsets,num_row,num_col,num_plane)
for s in range(0,num_subsets):
  
  i,j,k = get_ijk(s,num_row,num_col,num_plane)
  s_bounds = global_3d_subset_boundaries[s]
  
  if (k == 0):
    #Getting subsets in the above layer.
    subsets = [(i,j,k) for (i,j,k) in all_ijk if k==1]
    #Looping over above subsets.
    for n in range(0,len(subsets)):
      i,j,k = subsets[n]
      ss_id = (i*num_row+j) + k*(num_row*num_col)
      #Bounds of the potential neighbor.
      n_bounds = global_3d_subset_boundaries[ss_id]
      
      #Checking for overlap. If true, this into the adjacency matrix.
      overlap_bool = overlap(s_bounds,n_bounds)
      if (overlap_bool):
        adjacency_matrix_3d[s][ss_id] = 1
  
  #Checking for neighbors below only.
  elif (k == num_plane-1):
    subsets = [(i,j,k2) for (i,j,k2) in all_ijk if k2 == k-1]
    #Looping over below subsets.
    for n in range(0,len(subsets)):
      i,j,k = subsets[n]
      ss_id = (i*num_row+j) + k*(num_row*num_col)
      #Bounds of the potential neighbor.
      n_bounds = global_3d_subset_boundaries[ss_id]
      
      #Checking for overlap. If true, this into the adjacency matrix.
      overlap_bool = overlap(s_bounds,n_bounds)
      if (overlap_bool):
        adjacency_matrix_3d[s][ss_id] = 1

      