import numpy as np
import networkx as nx
from build_adjacency_matrix import build_graphs
from sweep_solver import make_edges_universal
from sweep_solver import add_conflict_weights
from sweep_solver import all_simple_paths_modified
from sweep_solver import nodes_being_solved_simple
from sweep_solver import match_delay_weights
from sweep_solver import modify_downstream_edges
import matplotlib.pyplot as plt
import time

plt.close("all")

numrow = 3
numcol = 3

adjacency_matrix = np.genfromtxt('matrices_2.csv',delimiter=",")

#A list that stores the time to solve each node.
time_to_solve = [1]*numrow*numcol

graphs = build_graphs(adjacency_matrix,numrow,numcol)
#Adding the uniform edge labels.
for g in range(0,4):
  graph = graphs[g]
  graph.add_weighted_edges_from((u,v,1) for u,v in graph.edges())

G,G1,G2,G3 = graphs

graphs = make_edges_universal(graphs)

plt.figure("G")
edge_labels_1 = nx.get_edge_attributes(G,'weight')
nx.draw(G,nx.spectral_layout(G,weight = None),with_labels = True)
nx.draw_networkx_edge_labels(G,nx.spectral_layout(G,weight = None),edge_labels=edge_labels_1)

G = graphs[0]

G = modify_downstream_edges(G,4,-1,[],1.0)

G = match_delay_weights(G)

plt.figure("G Post Delay")
edge_labels_1 = nx.get_edge_attributes(G,'weight')
nx.draw(G,nx.spectral_layout(G,weight = None),with_labels = True)
nx.draw_networkx_edge_labels(G,nx.spectral_layout(G,weight = None),edge_labels=edge_labels_1)