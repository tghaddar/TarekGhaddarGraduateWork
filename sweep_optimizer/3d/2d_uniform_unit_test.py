from copy import copy
from sweep_solver import plot_subset_boundaries
from sweep_solver import add_edge_cost
from sweep_solver import add_conflict_weights
from sweep_solver import compute_solve_time
from sweep_solver import print_simple_paths
from build_adjacency_matrix import build_adjacency
from flip_adjacency import flip_adjacency
import matplotlib.pyplot as plt
import numpy as np
from build_global_subset_boundaries import build_global_subset_boundaries
#from sweep_solver import 
import warnings
import networkx as nx
warnings.filterwarnings("ignore")

t_u = 1.0
m_l = 1.0
t_comm = 1.0
latency = 1.0
upc = 8.0
upbc = 4.0

plt.close("all")


#Number of cuts in the x direction.
N_x = 1
#Number of cuts in the y direction.
N_y = 1

#The subset boundaries.
x_cuts = [0.0, 5, 10.0]
y_cuts = [[0.0, 5, 10.0],[0.0, 5, 10.0]]

global_subset_boundaries = build_global_subset_boundaries(N_x,N_y,x_cuts,y_cuts)
adjacency_matrix = build_adjacency(global_subset_boundaries,N_x, N_y, y_cuts)

#Getting the upper triangular portion of the adjacency_matrix
adjacency_matrix_0 = np.triu(adjacency_matrix)
#Time to build the graph
G = nx.DiGraph(adjacency_matrix_0)
plt.figure("G")
nx.draw(G,with_labels = True)

#Test what lower triangular looks like
adjacency_matrix_3 = np.tril(adjacency_matrix)
G_3 = nx.DiGraph(adjacency_matrix_3)
plt.figure("G_3")
nx.draw(G_3,with_labels = True)

#To get the top left and bottom right quadrants, we have to reverse our ordering by column.
adjacency_flip,id_map = flip_adjacency(adjacency_matrix,N_y+1,N_x+1)
adjacency_matrix_1 = np.triu(adjacency_flip)
plt.figure("G_1")
G_1 = nx.DiGraph(adjacency_matrix_1)
G_1 = nx.relabel_nodes(G_1,id_map,copy=True)
nx.draw(G_1,with_labels = True)

#Bottom right quadrant.
adjacency_matrix_2 = np.tril(adjacency_flip)
plt.figure("G_2")
G_2 = nx.DiGraph(adjacency_matrix_2)
G_2 = nx.relabel_nodes(G_2,id_map,copy=True)
nx.draw(G_2,with_labels = True)

graphs = [G,G_1,G_2,G_3]

#Storing all simple paths for each graph.
all_simple_paths = []
for graph in graphs:
  copy_graph = copy(graph)
  start_node = [x for x in copy_graph.nodes() if copy_graph.in_degree(x) == 0][0]
  end_node = [x for x in copy_graph.nodes() if copy_graph.out_degree(x) == 0][0]
  simple_paths = nx.all_simple_paths(graph,start_node,end_node)
  all_simple_paths.append(simple_paths)
