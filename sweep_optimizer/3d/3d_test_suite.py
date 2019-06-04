# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:33:42 2019

@author: tghad
"""
from build_3d_adjacency import build_3d_global_subset_boundaries
from build_3d_adjacency import build_adjacency_matrix
from build_3d_adjacency import build_graphs
from sweep_solver import make_edges_universal
from sweep_solver import add_conflict_weights,pipeline_offset
from sweep_solver import compute_solve_time
from sweep_solver import get_max_incoming_weight
from mesh_processor import create_3d_cuts
import matplotlib.pyplot as plt 
import warnings
import networkx as nx
warnings.filterwarnings("ignore")

plt.close("all")


#Number of cuts in the x direction.
N_x = 2
#Number of cuts in the y direction.
N_y = 2
#Number of cuts in the z direction.
N_z = 2
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


num_angles = 1
unweighted = False

z_cuts,x_cuts,y_cuts = create_3d_cuts(global_x_min,global_x_max,num_col,global_y_min,global_y_max,num_row,global_z_min,global_z_max,num_plane)

global_subset_boundaries = build_3d_global_subset_boundaries(N_x,N_y,N_z,x_cuts,y_cuts,z_cuts)

adjacency_matrix = build_adjacency_matrix(x_cuts,y_cuts,z_cuts,num_row,num_col,num_plane)

graphs = build_graphs(adjacency_matrix,num_row,num_col,num_plane)

num_graphs = len(graphs)

#Adding the uniform edge labels.
for g in range(0,num_graphs):
  graph = graphs[g]
  graph.add_weighted_edges_from((u,v,1) for u,v in graph.edges())


graphs = pipeline_offset(graphs,num_angles,time_to_solve)
graphs = make_edges_universal(graphs)

#A list that stores the time to solve each node.
time_to_solve = [[1]*num_subsets for g in range(0,num_graphs)]

graphs = add_conflict_weights(graphs,time_to_solve,num_angles,unweighted)

time_to_node_all_octants = [None]*num_graphs
for g in range(0,num_graphs):
  G = graphs[g]
  time_to_node = {}
  for n in range(0,num_subsets):
    time = get_max_incoming_weight(G,n)
    if time in time_to_node:
     time_to_node[time] = time_to_node[time] + [n]
    else:
      time_to_node[time] = [n]
  
  time_to_node = dict(sorted(time_to_node.items(),key = lambda x: x[0]))
  time_to_node_all_octants[g] = time_to_node

f = open("octant_stage_example.txt","w")

for g in range(0,num_graphs):
  f.write("Octant: " + str(g) + "\n")
  time_to_node = time_to_node_all_octants[g]
  for key in time_to_node:
    f.write("Stage " + str(key) + ": ")
    nodes = time_to_node[key]
    for n in nodes:
      f.write(str(n) + " ")
    f.write("\n")
  f.write("\n")

f.close()
print(compute_solve_time(graphs))