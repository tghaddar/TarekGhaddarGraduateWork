import numpy as np
import networkx as nx
from build_adjacency_matrix import build_graphs
from sweep_solver import make_edges_universal
from sweep_solver import add_conflict_weights
from sweep_solver import all_simple_paths_modified
from sweep_solver import nodes_being_solved_simple
import matplotlib.pyplot as plt
import time


numrow = 2
numcol = 2

adjacency_matrix = np.genfromtxt('matrices_1.csv',delimiter=",")

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
nodes = nodes_being_solved_simple(G,[0],0.0001,time_to_solve)
end = time.time()
print(end-start)
print(nodes)