from sweep_solver import get_heaviest_path_simple
from build_adjacency_matrix import build_graphs
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sweep_solver import make_edges_universal_new
plt.close("all")

numrow = 2
numcol = 2 

adjacency_matrix = np.genfromtxt('matrices_1_random.csv',delimiter=",")


graphs = build_graphs(adjacency_matrix,numrow,numcol)
num_graphs = len(graphs)

#Adding the uniform edge labels.
for g in range(0,num_graphs):
  graph = graphs[g]
  graph.add_weighted_edges_from((u,v,1) for u,v in graph.edges())

G,G1,G2,G3 = graphs

heaviest_path_lengths = list(get_heaviest_path_simple(G2,2,1))
print(heaviest_path_lengths)

#graphs = make_edges_universal_new(graphs)

#plt.figure("G universal")
#edge_labels_1 = nx.get_edge_attributes(G,'weight')
#nx.draw(G,nx.spectral_layout(G),with_labels = True)
#nx.draw_networkx_edge_labels(G,nx.spectral_layout(G),edge_labels=edge_labels_1)
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
##
#plt.figure("G3 universal")
#edge_labels_1 = nx.get_edge_attributes(G3,'weight')
#nx.draw(G3,nx.spectral_layout(G3,weight = None),with_labels = True)
#nx.draw_networkx_edge_labels(G3,nx.spectral_layout(G3,weight = None),edge_labels=edge_labels_1)