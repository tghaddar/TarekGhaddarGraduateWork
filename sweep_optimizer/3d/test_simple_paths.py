import numpy as np
import networkx as nx
from build_adjacency_matrix import build_graphs
from sweep_solver import make_edges_universal
from sweep_solver import add_conflict_weights
from sweep_solver import all_simple_paths_modified
from sweep_solver import nodes_being_solved_faster
import matplotlib.pyplot as plt
import time


numrow = 20
numcol = 20

adjacency_matrix = np.genfromtxt('matrices_19.csv',delimiter=",")

#A list that stores the time to solve each node.
time_to_solve = [1]*numrow*numcol

graphs = build_graphs(adjacency_matrix,numrow,numcol)
#Adding the uniform edge labels.
for g in range(0,4):
  graph = graphs[g]
  graph.add_weighted_edges_from((u,v,1) for u,v in graph.edges())

G,G1,G2,G3 = graphs

graphs = make_edges_universal(graphs)

G = graphs[0]

start = time.time()
simple_paths = all_simple_paths_modified(G,0,-1,time_to_solve,cutoff = 2.0)
a = list(simple_paths)
#Making the list unique.
a = list(set(a))
a = sorted(a)
end = time.time()
print(end-start)