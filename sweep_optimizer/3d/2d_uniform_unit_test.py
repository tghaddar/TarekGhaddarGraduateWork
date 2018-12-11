from copy import copy
from sweep_solver import plot_subset_boundaries
from sweep_solver import add_edge_cost
from sweep_solver import add_conflict_weights
from sweep_solver import compute_solve_time
from sweep_solver import print_simple_paths
from sweep_solver import convert_generator
from build_adjacency_matrix import build_adjacency
from flip_adjacency_2d import flip_adjacency
from sweep_solver import get_heaviest_path
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
upc = 4.0
upbc = 2.0

plt.close("all")


#Number of cuts in the x direction.
N_x = 1
#Number of cuts in the y direction.
N_y = 1

num_row = N_y + 1
num_col = N_x + 1
num_plane = 0

#The subset boundaries.
step = (10.0)/num_col
x_cuts = [0.0,10.0]
y_cuts = []
for i in range(0,N_x):
  y_cuts.append([0,10.0])

for i in range(0,N_x):
  x_cuts.append((i+1)*step)
  for j in range(0,N_y):
    y_cuts[i].append((j+1)*step)
  y_cuts[i] = sorted(y_cuts[i])

y_cuts.append(y_cuts[0])
x_cuts = sorted(x_cuts)
    
cell_dist = [4096,5880,1000,10000]

global_subset_boundaries = build_global_subset_boundaries(N_x,N_y,x_cuts,y_cuts)
adjacency_matrix = build_adjacency(global_subset_boundaries,N_x, N_y, y_cuts)
plt.figure("2d_layer")
for i in range(0,4):
  
  subset_boundary = global_subset_boundaries[i]
  
  xmin = subset_boundary[0]
  xmax = subset_boundary[1]
  ymin = subset_boundary[2]
  ymax = subset_boundary[3]
  
  x = [xmin,xmax,xmax,xmin,xmin]
  y = [ymin,ymin,ymax,ymax,ymin]
  center_x = (xmin+xmax)/2
  center_y = (ymin+ymax)/2
  
  plt.plot(x,y,'b')
  plt.text(center_x,center_y,str(cell_dist[i]),fontsize=20)

plt.savefig("../../Prelim/figures/2d_layer.pdf")

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


num_subsets = (N_x+1)*(N_y+1)
#Equivalent number of cells per subset.

#for i in range(0,num_subsets):
#  cell_dist.append(4096.0)


num_total_cells = sum(cell_dist)

graphs = add_edge_cost(graphs,num_total_cells,global_subset_boundaries,cell_dist,t_u,upc,upbc,t_comm,latency,m_l,num_row,num_col,num_plane)

for ig in range(0,len(graphs)):
  for line in nx.generate_edgelist(graphs[ig],data=True):
    print(line)
  print("\n")

heaviest_paths_edge = []
for g in range(0,len(graphs)):
  copy_graph = copy(graphs[g])
  simple_paths = all_simple_paths[g]
  heaviest_path,heaviest_path_weight = get_heaviest_path(copy_graph,simple_paths)
  heaviest_paths_edge.append(heaviest_path)
  
  
#Storing all simple paths for each graph.
all_simple_paths = []
for graph in graphs:
  copy_graph = copy(graph)
  start_node = [x for x in copy_graph.nodes() if copy_graph.in_degree(x) == 0][0]
  end_node = [x for x in copy_graph.nodes() if copy_graph.out_degree(x) == 0][0]
  simple_paths = nx.all_simple_paths(graph,start_node,end_node)
  all_simple_paths.append(simple_paths)
  
#for ig in range(0,len(graphs)):
#  for line in nx.generate_edgelist(graphs[ig],data=True):
#    print(line)
#  print("\n")

graphs = add_conflict_weights(graphs,all_simple_paths,latency,cell_dist,num_row,num_col,num_plane)




all_graph_time,time,heaviest_paths = compute_solve_time(graphs,cell_dist,t_u,upc,global_subset_boundaries,num_row,num_col,num_plane)
print(all_graph_time)
for ig in range(0,len(graphs)):
  for line in nx.generate_edgelist(graphs[ig],data=True):
    print(line)
  print("\n")