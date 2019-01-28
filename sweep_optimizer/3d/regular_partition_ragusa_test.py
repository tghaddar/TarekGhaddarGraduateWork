import numpy as np
import networkx as nx
from build_adjacency_matrix import build_graphs
from sweep_solver import make_edges_universal
from sweep_solver import add_conflict_weights
import matplotlib.pyplot as plt
plt.close("all")

numrow = 4
numcol = 4

adjacency_matrix = np.genfromtxt('matrices.csv',delimiter=",")

graphs = build_graphs(adjacency_matrix,numrow,numcol)
num_graphs = len(graphs)

#Adding the uniform edge labels.
for g in range(0,num_graphs):
  graph = graphs[g]
  graph.add_weighted_edges_from((u,v,1) for u,v in graph.edges())

G = graphs[0]
plt.figure("Test for adding uniform edge weights")
edge_labels_1 = nx.get_edge_attributes(G,'weight')
nx.draw(G,nx.spectral_layout(G,weight = None),with_labels = True)
nx.draw_networkx_edge_labels(G,nx.spectral_layout(G,weight = None),edge_labels=edge_labels_1)


graphs = make_edges_universal(graphs)

plt.figure("Test for universal edge weights")
edge_labels_1 = nx.get_edge_attributes(G,'weight')
nx.draw(G,nx.spectral_layout(G,weight = None),with_labels = True)
nx.draw_networkx_edge_labels(G,nx.spectral_layout(G,weight = None),edge_labels=edge_labels_1)


#A list that stores the time to solve each node.
time_to_solve = [1]*16
graphs = add_conflict_weights(graphs,time_to_solve)