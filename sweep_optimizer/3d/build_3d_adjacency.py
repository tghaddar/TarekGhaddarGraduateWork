#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 10:41:34 2018

@author: tghaddar

Test space for 3D adjacency matrix building.
"""

from build_global_subset_boundaries import build_global_subset_boundaries
from build_adjacency_matrix import build_adjacency
from utilities import get_ijk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sweep_solver
import networkx as nx
import flip_adjacency
import warnings
warnings.filterwarnings("ignore") 

plt.close("all")

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
  if ( (s_topleft[0] >= n_botright[0]) or (n_topleft[0] >= s_botright[0]) ):
    return False
  
  #If one rectangle is above the other.
  if ( (s_topleft[1] <= n_botright[1]) or (n_topleft[1] <= s_botright[1]) ):
    return False
  
  return True




def build_adjacency_matrix(x_cuts,y_cuts,z_cuts,num_row,num_col,num_plane):
  
  num_subsets = num_row*num_col*num_plane
  num_subsets_2d = num_row*num_col

  #Adding the adjacency matrix for each layer.
  adjacency_matrix_3d = np.zeros((num_subsets,num_subsets))
  for z in range(0,num_plane):
    x_cuts_plane = x_cuts[z]
    y_cuts_plane = y_cuts[z]
    global_2d_subset_boundaries = build_global_subset_boundaries(num_col-1, num_row-1,x_cuts_plane,y_cuts_plane)
    adjacency_matrix = build_adjacency(global_2d_subset_boundaries,num_col-1,num_row-1,y_cuts_plane)
    
    adjacency_matrix_3d[z*num_subsets_2d:(z+1)*num_subsets_2d,z*num_subsets_2d:(z+1)*num_subsets_2d] = adjacency_matrix

  
  global_3d_subset_boundaries = build_3d_global_subset_boundaries(num_col-1,num_row-1,num_plane-1,x_cuts,y_cuts,z_cuts)
  
  #Time to add in the neighbors in 3D. We'll loop over the subsets in each layer and add in neighbors that are in the layer above and below.
  all_ijk = get_all_ijk(num_subsets,num_row,num_col,num_plane)
  for s in range(0,num_subsets):
    
    i,j,k = get_ijk(s,num_row,num_col,num_plane)
    s_bounds = global_3d_subset_boundaries[s]
    
    if (k == 0):
      #Getting subsets in the above layer.
      subsets = [(i,j,k) for (i,j,k) in all_ijk if k == 1]
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
    else:
      top_subsets = [(i,j,k2) for (i,j,k2) in all_ijk if k2 == k+1]
      bot_subsets = [(i,j,k2) for (i,j,k2) in all_ijk if k2 == k-1]
      
      #Looping over top and bottom layers for neighbors.
      for n in range(0,len(top_subsets)):
        i,j,k = top_subsets[n]
        ss_id = (i*num_row+j) + k*(num_row*num_col)
        #Bounds of the potential neighbor.
        n_bounds = global_3d_subset_boundaries[ss_id]
        
        #Checking for overlap. If true, this into the adjacency matrix.
        overlap_bool = overlap(s_bounds,n_bounds)
        if (overlap_bool):
          adjacency_matrix_3d[s][ss_id] = 1
      
      for n in range(0,len(bot_subsets)):
        i,j,k = bot_subsets[n]
        ss_id = (i*num_row+j) + k*(num_row*num_col)
        #Bounds of the potential neighbor.
        n_bounds = global_3d_subset_boundaries[ss_id]
        
        #Checking for overlap. If true, this into the adjacency matrix.
        overlap_bool = overlap(s_bounds,n_bounds)
        if (overlap_bool):
          adjacency_matrix_3d[s][ss_id] = 1

  return adjacency_matrix_3d

#Creating the graphs.
def build_graphs(adjacency_matrix_3d,num_row,num_col,num_plane):
  
  #Getting the upper triangular portion of the adjacency_matrix
  adjacency_matrix_0 = np.triu(adjacency_matrix_3d)
  #Time to build the graph for octant 0
  G = nx.DiGraph(adjacency_matrix_0)
  plt.figure(2)
  nx.draw(G,nx.shell_layout(G),with_labels = True)
  plt.savefig('digraph.pdf')
  
  #Lower triangular matrix.
  adjacency_matrix_7 = np.tril(adjacency_matrix_3d)
  #Building graph for octant 7
  G_7 = nx.DiGraph(adjacency_matrix_7)
  plt.figure(3)
  nx.draw(G_7,nx.shell_layout(G_7),with_labels = True)
  plt.savefig('digraph7.pdf')
  
  #Reordering for octant 1.
  adjacency_flip,id_map = flip_adjacency.flip_adjacency_1(adjacency_matrix_3d,num_row,num_col,num_plane)
  adjacency_matrix_1 = np.triu(adjacency_flip)
  G_1 = nx.DiGraph(adjacency_matrix_1)
  G_1 = nx.relabel_nodes(G_1,id_map,copy=True)
  plt.figure(4)
  nx.draw(G_1,nx.shell_layout(G_1),with_labels=True)
  plt.savefig('digraph1.pdf')
  
  adjacency_matrix_6 = np.tril(adjacency_flip)
  G_6 = nx.DiGraph(adjacency_matrix_6)
  G_6 = nx.relabel_nodes(G_6,id_map,copy=True)
  plt.figure(5)
  nx.draw(G_6,nx.shell_layout(G_6),with_labels=True)
  plt.savefig('digraph6.pdf')
  
  adjacency_flip,id_map = flip_adjacency.flip_adjacency_2(adjacency_matrix_3d,num_row,num_col,num_plane)
  adjacency_matrix_2 = np.triu(adjacency_flip)
  G_2 = nx.DiGraph(adjacency_matrix_2)
  G_2 = nx.relabel_nodes(G_2,id_map,copy=True)
  plt.figure(6)
  nx.draw(G_2,nx.shell_layout(G_2),with_labels=True)
  plt.savefig('digraph2.pdf')
  
  adjacency_matrix_5 = np.tril(adjacency_flip)
  G_5 = nx.DiGraph(adjacency_matrix_5)
  G_5 = nx.relabel_nodes(G_5,id_map,copy=True)
  plt.figure(7)
  nx.draw(G_5,nx.shell_layout(G_5),with_labels=True)
  plt.savefig('digraph5.pdf')
  
  adjacency_flip,id_map = flip_adjacency.flip_adjacency_3(adjacency_matrix_3d,num_row,num_col,num_plane)
  adjacency_matrix_3 = np.triu(adjacency_flip)
  G_3 = nx.DiGraph(adjacency_matrix_3)
  G_3 = nx.relabel_nodes(G_3,id_map,copy=True)
  plt.figure(8)
  nx.draw(G_3,nx.shell_layout(G_3),with_labels=True)
  plt.savefig('digraph3.pdf')
  
  adjacency_matrix_4 = np.tril(adjacency_flip)
  G_4 = nx.DiGraph(adjacency_matrix_4)
  G_4 = nx.relabel_nodes(G_4,id_map,copy=True)
  plt.figure(9)
  nx.draw(G_4,nx.shell_layout(G_4),with_labels=True)
  plt.savefig('digraph4.pdf')

  #Storing all the graphs in a list.
  graphs = [G,G_1,G_2,G_3,G_4,G_5,G_6,G_7]
  
  return graphs





