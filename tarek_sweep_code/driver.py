#importing the adjacency building capabilities.
from build_adjacency_matrix import build_adjacency
from build_global_subset_boundaries import build_global_subset_boundaries
from build_graph import build_graph
import random
import matplotlib.pyplot as plt
import networkx as nx
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#Number of cuts in x direction.
N_x = 2
#Number of cuts in y direction.
N_y = 2

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

global_subset_boundaries = build_global_subset_boundaries(N_x,N_y,x_cuts,y_cuts)
adjacency_matrix = build_adjacency(global_subset_boundaries,N_x,N_y,y_cuts)


plt.figure(1)

subset_centers = []

for i in range(0,num_subsets):
  subset_boundary = global_subset_boundaries[i]
  xmin = subset_boundary[0]
  xmax = subset_boundary[1]
  ymin = subset_boundary[2]
  ymax = subset_boundary[3]

  center_x = (xmin+xmax)/2
  center_y = (ymin+ymax)/2

  subset_centers.append([center_x, center_y])

  x = [xmin, xmax, xmax, xmin, xmin]
  y = [ymin, ymin, ymax, ymax, ymin]

  plt.plot(x,y,'b')


for i in range (0, num_subsets):

  neighbors = adjacency_matrix[i]

  for j in range(0, len(neighbors)):
    n = neighbors[j]

    x = [subset_centers[i][0], subset_centers[n][0]]
    y = [subset_centers[i][1], subset_centers[n][1]]
    plt.plot(x,y,'r-o')

plt.savefig('adjacency_matrix.pdf')

#The adjacency matrix has been computed and plotted.
#Time to build the graph
G = build_graph(adjacency_matrix)

plt.figure(2)
nx.draw(G,with_labels = True)
plt.savefig('digraph.pdf')
