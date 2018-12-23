#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 16:46:26 2018
This code tests dfs algorithms that I may want to use.

@author: tghaddar
"""
import networkx as nx
import matplotlib.pyplot as plt
from copy import copy
from sweep_solver import make_edges_universal
from sweep_solver import nodes_being_solved

#Number of cuts in x.
N_x = 2
#Number of cuts in y.
N_y = 2

num_nodes = 9
plt.close("all")

G = nx.DiGraph()
for n in range(0,num_nodes):
  G.add_node(n)
  

#Node 0 edges.
G.add_edge(0,1,weight = 3)
G.add_edge(0,3,weight = 3)

#Node 1 edges.
G.add_edge(1,2,weight = 10)
G.add_edge(1,4,weight = 10)

#Node 2 edges.
G.add_edge(2,5,weight = 7)

#Node 3 edges.
G.add_edge(3,6,weight = 5)
G.add_edge(3,4,weight = 5)

#Node 4 edges.
G.add_edge(4,5,weight = 8)
G.add_edge(4,7,weight = 8)

#Node 5 edges.
G.add_edge(5,8,weight = 9)

#Node 6 edges.
G.add_edge(6,7,weight = 20)

#Node 7 edge.
G.add_edge(7,8,weight = 15)

plt.figure("Graph 0 Pre Universal Time")
edge_labels_1 = nx.get_edge_attributes(G,'weight')
nx.draw(G,nx.shell_layout(G),with_labels = True)
nx.draw_networkx_edge_labels(G,nx.shell_layout(G),edge_labels=edge_labels_1)

#Putting universal times on the weights.
graphs = [G]

graphs = make_edges_universal(graphs)

G = graphs[0]


plt.figure("Graph 0 Post Universal Time")
edge_labels_1 = nx.get_edge_attributes(G,'weight')
nx.draw(G,nx.shell_layout(G),with_labels = True)
nx.draw_networkx_edge_labels(G,nx.shell_layout(G),edge_labels=edge_labels_1)

##Testing our weight-based traversal.
#We try an arbitrary weight limit. 
weight_limit = 20.0

current_nodes = nodes_being_solved(G,weight_limit)
print(current_nodes)


