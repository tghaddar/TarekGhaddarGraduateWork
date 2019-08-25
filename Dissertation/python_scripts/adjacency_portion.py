import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
#sys.path.append(r'C:\Users\tghad\Documents\GitHub\TarekGhaddarGraduateWork\sweep_optimizer\3d')
from mesh_processor import create_2d_cuts
from sweep_solver import plot_subset_boundaries_2d
from build_global_subset_boundaries import build_global_subset_boundaries
from build_adjacency_matrix import build_adjacency,build_graphs 
import networkx as nx
plt.close("all")


#Pulling cut line data.
x_cuts_lbd = np.genfromtxt("x_cuts_5_worst.csv",delimiter=",")
y_cuts_lbd = np.genfromtxt("y_cuts_5_worst.csv",delimiter=",")

num_row = 5
num_col = 5
num_subsets = num_row*num_col
boundaries_lbd = build_global_subset_boundaries(num_col-1,num_row-1,x_cuts_lbd,y_cuts_lbd)

#plot_subset_boundaries_2d(boundaries_lbd,num_subsets,"../../figures/boundaries_worst.pdf")

adjacency_matrix = build_adjacency(boundaries_lbd,num_col-1,num_row-1,y_cuts_lbd)

f = open("first_adjacency_matrix.txt",'w')
f.write('$\\begin{pmatrix}\n')
#Looping over the adjacency matrix and writing it to latex.
for i in range(0,num_subsets):
  for j in range(0,num_subsets):
    f.write(str(int(adjacency_matrix[i][j])))
    if j < num_subsets-1:
      f.write('&')
  
  f.write('\\\ \n')


f.write('\\end{pmatrix}$\n')
f.close()

#Getting the upper triangular portion of the adjacency_matrix
adjacency_matrix_0 = np.triu(adjacency_matrix)
f = open("first_adjacency_matrix_ut.txt",'w')
f.write('$\\begin{pmatrix}\n')
#Looping over the adjacency matrix and writing it to latex.
for i in range(0,num_subsets):
  for j in range(0,num_subsets):
    f.write(str(int(adjacency_matrix_0[i][j])))
    if j < num_subsets-1:
      f.write('&')  
  f.write('\\\ \n')
f.write('\\end{pmatrix}$\n')
f.close()

#Getting the lower triangular portion of the adjacency_matrix
adjacency_matrix_3 = np.tril(adjacency_matrix)
f = open("first_adjacency_matrix_lt.txt",'w')
f.write('$\\begin{pmatrix}\n')
#Looping over the adjacency matrix and writing it to latex.
for i in range(0,num_subsets):
  for j in range(0,num_subsets):
    f.write(str(int(adjacency_matrix_3[i][j])))
    if j < num_subsets-1:
      f.write('&') 
  f.write('\\\ \n')
f.write('\\end{pmatrix}$\n')
f.close()

#Digraph plot for Q0.
plt.figure()
#Time to build the graph for Q0.
G = nx.DiGraph(adjacency_matrix_0)
nx.draw(G,nx.kamada_kawai_layout(G),with_labels=True)
plt.savefig("../../figures/25_graph0.pdf")


plt.figure()
G3 = nx.DiGraph(adjacency_matrix_3)
nx.draw(G3,nx.kamada_kawai_layout(G3),with_labels=True)
plt.savefig("../../figures/25_graph3.pdf")