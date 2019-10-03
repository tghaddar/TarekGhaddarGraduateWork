import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
#sys.path.append(r'C:\Users\tghad\Documents\GitHub\TarekGhaddarGraduateWork\sweep_optimizer\3d')
from mesh_processor import create_2d_cuts
from sweep_solver import plot_subset_boundaries_2d, get_node_positions
from build_global_subset_boundaries import build_global_subset_boundaries
from build_adjacency_matrix import build_adjacency,build_graphs 
from flip_adjacency_2d import flip_adjacency
import networkx as nx
plt.close("all")


#Pulling cut line data.
x_cuts_lbd = np.genfromtxt("x_cuts_3.csv",delimiter=",")
y_cuts_lbd = np.genfromtxt("y_cuts_3.csv",delimiter=",")

num_row = 3
num_col = 3
num_subsets = num_row*num_col
boundaries_lbd = build_global_subset_boundaries(num_col-1,num_row-1,x_cuts_lbd,y_cuts_lbd)

#plot_subset_boundaries_2d(boundaries_lbd,num_subsets,0,"../../figures/boundaries_worst.pdf")

adjacency_matrix = build_adjacency(boundaries_lbd,num_col-1,num_row-1,y_cuts_lbd)

#f = open("first_adjacency_matrix.txt",'w')
#f.write('$\\begin{pmatrix}\n')
##Looping over the adjacency matrix and writing it to latex.
#for i in range(0,num_subsets):
#  for j in range(0,num_subsets):
#    f.write(str(int(adjacency_matrix[i][j])))
#    if j < num_subsets-1:
#      f.write('&')
#  
#  f.write('\\\ \n')
#
#
#f.write('\\end{pmatrix}$\n')
#f.close()
#
##Getting the upper triangular portion of the adjacency_matrix
adjacency_matrix_0 = np.triu(adjacency_matrix)
#f = open("first_adjacency_matrix_ut.txt",'w')
#f.write('$\\begin{pmatrix}\n')
##Looping over the adjacency matrix and writing it to latex.
#for i in range(0,num_subsets):
#  for j in range(0,num_subsets):
#    f.write(str(int(adjacency_matrix_0[i][j])))
#    if j < num_subsets-1:
#      f.write('&')  
#  f.write('\\\ \n')
#f.write('\\end{pmatrix}$\n')
#f.close()
#
##Getting the lower triangular portion of the adjacency_matrix
adjacency_matrix_3 = np.tril(adjacency_matrix)
#f = open("first_adjacency_matrix_lt.txt",'w')
#f.write('$\\begin{pmatrix}\n')
##Looping over the adjacency matrix and writing it to latex.
#for i in range(0,num_subsets):
#  for j in range(0,num_subsets):
#    f.write(str(int(adjacency_matrix_3[i][j])))
#    if j < num_subsets-1:
#      f.write('&') 
#  f.write('\\\ \n')
#f.write('\\end{pmatrix}$\n')
#f.close()
#
Q = get_node_positions(boundaries_lbd,num_row,num_col)
##Digraph plot for Q0.
plt.figure()
#Time to build the graph for Q0.
G = nx.DiGraph(adjacency_matrix_0)
nx.draw(G,Q[0],with_labels=True,node_size=900,font_size=15,arrowsize=20)
plt.savefig("../../figures/9_graph0.pdf")
#
#
plt.figure()
G3 = nx.DiGraph(adjacency_matrix_3)
nx.draw(G3,Q[3],with_labels=True,node_size=900,font_size=15,arrowsize=20)
plt.savefig("../../figures/9_graph3.pdf")
#
#
##The flipped adjacency matrix.
adjacency_flip,id_map = flip_adjacency(adjacency_matrix,num_row,num_col)
#f = open("flipped_adjacency_matrix.txt",'w')
#f.write('$\\begin{pmatrix}\n')
##Looping over the adjacency matrix and writing it to latex.
#for i in range(0,num_subsets):
#  for j in range(0,num_subsets):
#    f.write(str(int(adjacency_flip[i][j])))
#    if j < num_subsets-1:
#      f.write('&')
#  
#  f.write('\\\ \n')
#
#
#f.write('\\end{pmatrix}$\n')
#f.close()
#Plotting the flipped subset ordering.
#plot_subset_boundaries_2d(boundaries_lbd,num_subsets,id_map,"../../figures/boundaries_worst_flipped.pdf")
#
#
#Quadrant 1
adjacency_matrix_1 = np.triu(adjacency_flip)
G1 = nx.DiGraph(adjacency_matrix_1)
G1 = nx.relabel_nodes(G1,id_map,copy=True)
plt.figure()
nx.draw(G1,Q[1],with_labels=True,node_size=900,font_size=15,arrowsize=20)
plt.savefig("../../figures/9_graph1.pdf")
plt.close()
#
#Bottom right quadrant.
adjacency_matrix_2 = np.tril(adjacency_flip)
G2 = nx.DiGraph(adjacency_matrix_2)
G2 = nx.relabel_nodes(G2,id_map,copy=True)
plt.figure()
nx.draw(G2,Q[2],with_labels=True,node_size=900,font_size=15,arrowsize=20)
plt.savefig("../../figures/9_graph2.pdf")
plt.close()

