import numpy as np
import networkx as nx
from build_adjacency_matrix import build_graphs
from sweep_solver import make_edges_universal
from sweep_solver import add_conflict_weights
from sweep_solver import compute_solve_time
from mesh_processor import create_2d_cuts
from sweep_solver import plot_subset_boundaries_2d
from build_global_subset_boundaries import build_global_subset_boundaries
import matplotlib.pyplot as plt
import time
plt.close("all")

start_tos = time.time()

numrow = 3
numcol = 3

adjacency_matrix = np.genfromtxt('matrices_2.csv',delimiter=",")

x_cuts,y_cuts = create_2d_cuts(0.0,10.0,3,0.0,10.0,3)
boundaries = build_global_subset_boundaries(2,2,x_cuts,y_cuts)
plot_subset_boundaries_2d(boundaries,9)


graphs = build_graphs(adjacency_matrix,numrow,numcol)
num_graphs = len(graphs)

#Adding the uniform edge labels.
for g in range(0,num_graphs):
  graph = graphs[g]
  graph.add_weighted_edges_from((u,v,1) for u,v in graph.edges())

G,G1,G2,G3 = graphs

plt.figure("G pre universal")
edge_labels_1 = nx.get_edge_attributes(G,'weight')
nx.draw(G,nx.spectral_layout(G,weight = None),with_labels = True)
nx.draw_networkx_edge_labels(G,nx.spectral_layout(G,weight = None),edge_labels=edge_labels_1)
plt.savefig("G_pre_universal.pdf")

start_before_universal = time.time()

graphs = make_edges_universal(graphs)

start_after_universal = time.time()
plt.figure("G universal")
edge_labels_1 = nx.get_edge_attributes(G,'weight')
nx.draw(G,nx.spectral_layout(G,weight=None),with_labels = True)
nx.draw_networkx_edge_labels(G,nx.spectral_layout(G,weight = None),edge_labels=edge_labels_1)
plt.savefig("G_universal.pdf")
#
#plt.figure("G1 universal")
#edge_labels_1 = nx.get_edge_attributes(G1,'weight')
#nx.draw(G1,nx.spectral_layout(G1),with_labels = True)
#nx.draw_networkx_edge_labels(G1,nx.spectral_layout(G1),edge_labels=edge_labels_1)
#
#plt.figure("G2 universal")
#edge_labels_1 = nx.get_edge_attributes(G2,'weight')
#nx.draw(G2,nx.spectral_layout(G2),with_labels = True)
#nx.draw_networkx_edge_labels(G2,nx.spectral_layout(G2),edge_labels=edge_labels_1)
#
#plt.figure("G3 universal")
#edge_labels_1 = nx.get_edge_attributes(G3,'weight')
#nx.draw(G3,nx.spectral_layout(G3,weight = None),with_labels = True)
#nx.draw_networkx_edge_labels(G3,nx.spectral_layout(G3,'weight'),edge_labels=edge_labels_1)



#A list that stores the time to solve each node.
time_to_solve = [1]*numrow*numcol

graphs = add_conflict_weights(graphs,time_to_solve)
#
#end_tos = time.time()
#print("Universal Edges: ", start_after_universal - start_before_universal)
#print("Post Universal Edges: ", end_tos - start_after_universal)
#print("Total Run Time: ", end_tos - start_tos)
#
#all_times,max_time = compute_solve_time(graphs)
#print("Final Solve Time: ",max_time)

plt.figure("G")
edge_labels_1 = nx.get_edge_attributes(G,'weight')
nx.draw(G,nx.spectral_layout(G,weight = None),with_labels = True)
nx.draw_networkx_edge_labels(G,nx.spectral_layout(G,weight = None),edge_labels=edge_labels_1)


plt.figure("G1 universal")
edge_labels_1 = nx.get_edge_attributes(G1,'weight')
nx.draw(G1,nx.spectral_layout(G1),with_labels = True)
nx.draw_networkx_edge_labels(G1,nx.spectral_layout(G1),edge_labels=edge_labels_1)

plt.figure("G2 universal")
edge_labels_1 = nx.get_edge_attributes(G2,'weight')
nx.draw(G2,nx.spectral_layout(G2),with_labels = True)
nx.draw_networkx_edge_labels(G2,nx.spectral_layout(G2),edge_labels=edge_labels_1)

plt.figure("G3")
edge_labels_1 = nx.get_edge_attributes(G3,'weight')
nx.draw(G3,nx.spectral_layout(G3),with_labels = True)
nx.draw_networkx_edge_labels(G3,nx.spectral_layout(G3),edge_labels=edge_labels_1)
