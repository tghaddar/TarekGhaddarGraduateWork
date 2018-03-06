#importing the adjacency building capabilities.
from build_adjacency_matrix import build_adjacency
from build_global_subset_boundaries import build_global_subset_boundaries
from flip_adjacency import flip_adjacency
from perform_sweep import perform_sweep
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import warnings
warnings.filterwarnings("ignore") 

plt.close("all")

#Number of cuts in x direction.
N_x = 1
#Number of cuts in y direction.
N_y = 1

if (N_y > N_x):
  raise Exception('N_x >= N_y because our model assumes P_x >= P_y' )

#Number of subsets
num_subsets = (N_x+1)*(N_y+1)

#Domain global bounds
global_x_min = 0.0
global_x_max = 10.0
global_y_min = 0.0
global_y_max = 10.0

#Stores x and y cuts
x_cuts = []
y_cuts = []

#Populating x and y cuts
for i in range(0,N_x):
  cut = random.uniform(global_x_min,global_x_max)
  x_cuts.append(cut)
  

  #For each column, we need uniformly distributed random values of y_cuts.
  y_cuts_column = []
  for j in range(0,N_y):
    cut_y = random.uniform(global_y_min,global_y_max)
    y_cuts_column.append(cut_y)

  y_cuts_column.append(global_y_min)
  y_cuts_column.append(global_y_max)
  y_cuts_column.sort()
  y_cuts.append(y_cuts_column)

x_cuts.append(global_x_min)
x_cuts.append(global_x_max)
x_cuts.sort()
#Last column
y_cuts_column = []
for i in range(0,N_y):
  cut_y = random.uniform(global_y_min, global_y_max)
  y_cuts_column.append(cut_y)

y_cuts_column.append(global_y_min)
y_cuts_column.append(global_y_max)
y_cuts_column.sort()
y_cuts.append(y_cuts_column)

#Test Case
x_cuts[1] = 5
y_cuts[0][1] = 3
y_cuts[1][1] = 7

global_subset_boundaries = build_global_subset_boundaries(N_x,N_y,x_cuts,y_cuts)
adjacency_matrix = build_adjacency(global_subset_boundaries,N_x,N_y,y_cuts)

#Getting the upper triangular portion of the adjacency_matrix
adjacency_matrix_0 = np.triu(adjacency_matrix)
#Time to build the graph
G = nx.DiGraph(adjacency_matrix_0)
plt.figure(2)
nx.draw(G,with_labels = True)
plt.savefig('digraph.pdf')

#Test what lower triangular looks like
adjacency_matrix_3 = np.tril(adjacency_matrix)
G_3 = nx.DiGraph(adjacency_matrix_3)
plt.figure(3)
nx.draw(G_3,with_labels = True)
plt.savefig('digraph3.pdf')

#To get the top left and bottom right quadrants, we have to reverse our ordering by column.
adjacency_flip,id_map = flip_adjacency(adjacency_matrix,N_y+1,N_x+1)
adjacency_matrix_1 = np.triu(adjacency_flip)
G_1 = nx.DiGraph(adjacency_matrix_1)
G_1 = nx.relabel_nodes(G_1,id_map,copy=True)
plt.figure(4)
nx.draw(G_1,with_labels=True)
plt.savefig('digraph1.pdf')

#Bottom right quadrant.
adjacency_matrix_2 = np.tril(adjacency_flip)
G_2 = nx.DiGraph(adjacency_matrix_2)
G_2 = nx.relabel_nodes(G_2,id_map,copy=True)
plt.figure(5)
nx.draw(G_2,with_labels=True)
plt.savefig('digraph2.pdf')

all_graphs = [G,G_1,G_2,G_3]

#Angles per quadrant.
n_angle = 1
#Number of quadrants.
n_quad = 4
#Performing the sweep.
n_stages = perform_sweep(all_graphs,n_angle)

num = G.successors(0)
print(G.predecessors())




