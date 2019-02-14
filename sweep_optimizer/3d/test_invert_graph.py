#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:51:31 2019

@author: tghaddar
"""

import networkx as nx
import matplotlib.pyplot as plt
from sweep_solver import invert_weights
from copy import deepcopy

plt.close("all")

G = nx.DiGraph()
for n in range(0,4):
  G.add_node(n)

G.add_edge(0,1,weight = 1)
G.add_edge(1,3,weight = 1/2)
G.add_edge(0,2,weight = 1)
G.add_edge(2,3,weight = 1/3)

graph_copy = deepcopy(G)

plt.figure("G")
edge_labels_1 = nx.get_edge_attributes(G,'weight')
nx.draw(G,nx.spectral_layout(G),with_labels = True)
nx.draw_networkx_edge_labels(G,nx.spectral_layout(G),edge_labels=edge_labels_1)

graph_copy = invert_weights(graph_copy)

#Getting the shortest path of the copy (longest path of the original graph).
longest_path = nx.shortest_path(G,0,3)

plt.figure("G_invert")
edge_labels_1 = nx.get_edge_attributes(graph_copy,'weight')
nx.draw(graph_copy,nx.spectral_layout(graph_copy),with_labels = True)
nx.draw_networkx_edge_labels(graph_copy,nx.spectral_layout(graph_copy),edge_labels=edge_labels_1)