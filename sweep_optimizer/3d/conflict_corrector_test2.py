#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 13:40:43 2018
A different method of correcting conflicts with 4 paths.
@author: tghaddar
"""

import networkx as nx
import matplotlib.pyplot as plt
import warnings
from sweep_solver import add_conflict_weights
from sweep_solver import get_heaviest_path
from sweep_solver import get_DOG
warnings.filterwarnings("ignore", category=DeprecationWarning)

plt.close("all")

G = nx.DiGraph()
G2 = nx.DiGraph()
latency = 4110.0

num_nodes = 5

for n in range(0,num_nodes):
  G.add_node(n)
  G2.add_node(n)

G.add_edge(0,1,weight = 3)
G.add_edge(1,2,weight = 10)
G.add_edge(2,3,weight = 2)
G.add_edge(3,4,weight = 8)
plt.figure("Graph 1 Pre Conflict")
edge_labels_1 = nx.get_edge_attributes(G,'weight')
nx.draw(G,nx.shell_layout(G),with_labels = True)
nx.draw_networkx_edge_labels(G,nx.shell_layout(G),edge_labels=edge_labels_1)

G2.add_edge(4,3,weight = 1)
G2.add_edge(3,2,weight = 8)
G2.add_edge(2,1,weight = 2)
G2.add_edge(1,0,weight = 10)
edge_labels_2 = nx.get_edge_attributes(G2,'weight')
plt.figure("Graph 2 Pre Conflict")
nx.draw(G2,nx.shell_layout(G2),with_labels = True)
nx.draw_networkx_edge_labels(G2,nx.shell_layout(G2),edge_labels=edge_labels_2)

path1 = nx.all_simple_paths(G,0,4)
path2 = nx.all_simple_paths(G2,4,0)

graphs = [G,G2]
paths = [path1,path2]

graphs = add_conflict_weights(graphs,paths,latency)

G = graphs[0]
G2 = graphs[1]

for line in nx.generate_edgelist(G,data=True):
  print(line)

print("\n")
print("G2\n")
for line in nx.generate_edgelist(G2,data=True):
  print(line)