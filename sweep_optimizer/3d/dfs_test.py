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
from sweep_solver import add_conflict_weights
from sweep_solver import find_next_interaction
from sweep_solver import find_conflicts
from sweep_solver import find_first_conflict

#Number of cuts in x.
N_x = 2
#Number of cuts in y.
N_y = 2

num_nodes = 9
plt.close("all")

G = nx.DiGraph()
G1 = nx.DiGraph()
G2 = nx.DiGraph()
G3 = nx.DiGraph()
for n in range(0,num_nodes):
  G.add_node(n)
  G1.add_node(n)
  G2.add_node(n)
  G3.add_node(n)
  

#Node 0 edges.
G.add_edge(0,1,weight = 3)
G.add_edge(0,3,weight = 3)
G2.add_edge(0,-1,weight = 3)
G1.add_edge(0,3,weight = 3)
G3.add_edge(0,1,weight = 3)

#Node 1 edges.
G.add_edge(1,2,weight = 10)
G.add_edge(1,4,weight = 10)
G1.add_edge(1,4,weight = 10)
G1.add_edge(1,0,weight = 10)
G2.add_edge(1,0,weight = 10)
G3.add_edge(1,2,weight = 10)

#Node 2 edges.
G.add_edge(2,5,weight = 7)
G1.add_edge(2,1,weight = 7)
G1.add_edge(2,5,weight = 7)
G2.add_edge(2,1,weight = 7)
G3.add_edge(2,-1,weight = 7)

#Node 3 edges.
G.add_edge(3,6,weight = 5)
G.add_edge(3,4,weight = 5)
G1.add_edge(3,6,weight = 5)
G2.add_edge(3,0,weight = 5)
G3.add_edge(3,0,weight = 5)
G3.add_edge(3,4,weight = 5)

#Node 4 edges.
G.add_edge(4,5,weight = 8)
G.add_edge(4,7,weight = 8)
G1.add_edge(4,3,weight = 8)
G1.add_edge(4,7,weight = 8)
G2.add_edge(4,1,weight = 8)
G2.add_edge(4,3,weight = 8)
G3.add_edge(4,1,weight = 8)
G3.add_edge(4,5,weight = 8)

#Node 5 edges.
G.add_edge(5,8,weight = 9)
G1.add_edge(5,4,weight = 9)
G1.add_edge(5,8,weight = 9)
G2.add_edge(5,2,weight = 9)
G2.add_edge(5,4,weight = 9)
G3.add_edge(5,2,weight = 9)

#Node 6 edges.
G.add_edge(6,7,weight = 20)
G1.add_edge(6,-1,weight = 20)
G2.add_edge(6,3,weight = 20)
G3.add_edge(6,3,weight = 20)
G3.add_edge(6,7,weight = 20)

#Node 7 edge.
G.add_edge(7,8,weight = 15)
G1.add_edge(7,6,weight = 15)
G2.add_edge(7,4,weight = 15)
G2.add_edge(7,6,weight = 15)
G3.add_edge(7,4,weight = 15)
G3.add_edge(7,8,weight = 15)

#Node 8 edge.
G.add_edge(8,-1,weight = 1)
G1.add_edge(8,7,weight = 1)
G2.add_edge(8,5,weight = 1)
G2.add_edge(8,7,weight = 1)
G3.add_edge(8,5,weight = 1)

#Putting universal times on the weights.
graphs = [G,G1,G2,G3]

graphs = make_edges_universal(graphs)

G = graphs[0]

plt.figure("Graph 0 Post Universal Time")
edge_labels_1 = nx.get_edge_attributes(G,'weight')
nx.draw(G,nx.shell_layout(G),with_labels = True)
nx.draw_networkx_edge_labels(G,nx.shell_layout(G),edge_labels=edge_labels_1)

plt.figure("Graph 1 Post Universal Time")
edge_labels_1 = nx.get_edge_attributes(G1,'weight')
nx.draw(G1,nx.shell_layout(G1),with_labels = True)
nx.draw_networkx_edge_labels(G1,nx.shell_layout(G1),edge_labels=edge_labels_1)

plt.figure("Graph 2 Post Universal Time")
edge_labels_1 = nx.get_edge_attributes(G2,'weight')
nx.draw(G2,nx.shell_layout(G2),with_labels = True)
nx.draw_networkx_edge_labels(G2,nx.shell_layout(G2),edge_labels=edge_labels_1)


plt.figure("Graph 3 Post Universal Time")
edge_labels_1 = nx.get_edge_attributes(G3,'weight')
nx.draw(G3,nx.shell_layout(G3),with_labels = True)
nx.draw_networkx_edge_labels(G3,nx.shell_layout(G3),edge_labels=edge_labels_1)
##Testing our weight-based traversal.
#We try an arbitrary weight limit. 
weight_limit = 14.0

current_nodes = []

for g in range(0,len(graphs)):
  
  current_nodes.append(nodes_being_solved(graphs[g],weight_limit))


#Testing the next interaction function.
next_time = find_next_interaction(graphs,0.0)
conflicting_nodes = find_conflicts(current_nodes)

first_node = find_first_conflict(conflicting_nodes,graphs)

#graphs = add_conflict_weights(graphs)
