import numpy as np
import networkx as nx
from build_adjacency_matrix import build_graphs
from sweep_solver import make_edges_universal
from sweep_solver import modify_downstream_edges
import matplotlib.pyplot as plt
import time

numrow = 10
numcol = 10

adjacency_matrix = np.genfromtxt('matrices_9.csv',delimiter=",")

graphs = build_graphs(adjacency_matrix,numrow,numcol)
num_graphs = len(graphs)

#Adding the uniform edge labels.
for g in range(0,num_graphs):
  graph = graphs[g]
  graph.add_weighted_edges_from((u,v,1) for u,v in graph.edges())

G,G1,G2,G3 = graphs

graphs = make_edges_universal(graphs)




downstream_nodes = nx.descendants(G,0)
modified_edges = []

delay = 1.0
start_loop = time.time()
for node in downstream_nodes:
  
  #Getting incoming edges to this node.
  in_edges = G.in_edges(node)
  
  for u,v in in_edges:
    G[u][v]['weight'] += delay
    modified_edges.append((u,v))

end_loop = time.time()
print("loop: ", end_loop - start_loop)


modified_edges = []
start_func = time.time()
G = modify_downstream_edges(G,0,-1,modified_edges,delay)
end_func = time.time()
print("function: ",end_func - start_func)
