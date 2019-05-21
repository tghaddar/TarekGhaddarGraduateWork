# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:33:42 2019

@author: tghad
"""
from build_3d_adjacency import build_3d_global_subset_boundaries
from build_3d_adjacency import build_adjacency_matrix
from build_3d_adjacency import build_graphs
from sweep_solver import make_edges_universal
from sweep_solver import add_conflict_weights
from sweep_solver import compute_solve_time
from mesh_processor import create_3d_cuts
import matplotlib.pyplot as plt 
import warnings
import networkx as nx
warnings.filterwarnings("ignore")

plt.close("all")


#Number of cuts in the x direction.
N_x = 3
#Number of cuts in the y direction.
N_y = 3
#Number of cuts in the z direction.
N_z = 3
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
step_z = float((global_z_max - global_z_min)/(N_z+1))
step_y = float((global_y_max - global_y_min)/(N_y+1))
step_x = float((global_x_max - global_x_min)/(N_x+1))

z_cuts,x_cuts,y_cuts = create_3d_cuts(global_x_min,global_x_max,num_col,global_y_min,global_y_max,num_row,global_z_min,global_z_max,num_plane)

global_subset_boundaries = build_3d_global_subset_boundaries(N_x,N_y,N_z,x_cuts,y_cuts,z_cuts)

adjacency_matrix = build_adjacency_matrix(x_cuts,y_cuts,z_cuts,num_row,num_col,num_plane)

graphs = build_graphs(adjacency_matrix,num_row,num_col,num_plane)

num_graphs = len(graphs)

#Adding the uniform edge labels.
for g in range(0,num_graphs):
  graph = graphs[g]
  graph.add_weighted_edges_from((u,v,1) for u,v in graph.edges())


graphs = make_edges_universal(graphs)


#for g in range(0,num_graphs):
#  plt.figure(str(g))
#  edge_labels_1 = nx.get_edge_attributes(graphs[g],'weight')
#  if g < 3:
#    nx.draw(graphs[g],nx.spectral_layout(graphs[g]),with_labels = True)
#    nx.draw_networkx_edge_labels(graphs[g],nx.spectral_layout(graphs[g]),edge_labels=edge_labels_1)
#  else:
#    nx.draw(graphs[g],nx.spectral_layout(graphs[g],weight=100),with_labels = True)
#    nx.draw_networkx_edge_labels(graphs[g],nx.spectral_layout(graphs[g],weight=100),edge_labels=edge_labels_1)
#  plt.savefig("graph_"+str(g)+".pdf")
#  plt.close()

#A list that stores the time to solve each node.
time_to_solve = [[1]*num_subsets for g in range(0,num_graphs)]

graphs = add_conflict_weights(graphs,time_to_solve)

print(compute_solve_time(graphs))