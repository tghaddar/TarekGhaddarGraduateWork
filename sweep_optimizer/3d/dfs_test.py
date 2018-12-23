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
from sweep_solver import get_weight_sum
from sweep_solver import get_heaviest_path

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

#We need to find the number of depth levels.

#A copy of the graph.
copy_graph = copy(G)
#Getting the starting node.
start_node = [x for x in copy_graph.nodes() if copy_graph.in_degree(x) == 0][0]
end_node = [x for x in copy_graph.nodes() if copy_graph.out_degree(x) == 0][0]

#A list storing the heaviest path length to each node.
heavy_path_lengths = [None]*num_nodes
#Looping over nodes to get the longest path to each node.
for n in range(0,num_nodes):
  
  #Getting all simple paths to the node.
  simple_paths = nx.all_simple_paths(G,start_node,n)
  #The heaviest path and the length of the heaviest path.
  heaviest_path,heaviest_path_length = get_heaviest_path(G,simple_paths)
  
  #Storing this value in heavy_path_lengths.
  heavy_path_lengths[n] = heaviest_path_length
  
#Storing the heavy path lengths as the weight value to all preceding edges.
for n in range(0,num_nodes):
  
  #The starting node has no preceding edges so we skip it.
  if (n != start_node):
    #Getting the weight we want for preceding edges.
    new_weight = heavy_path_lengths[n]
    #Getting the predecessors to this node in the graph.
    predecessors = list(G.predecessors(n))
    num_pred = len(predecessors)
    for p in range(0,num_pred):
      pred = predecessors[p]
      G[pred][n]['weight'] = new_weight
    

plt.figure("Graph 0 Post Universal ish Time")
edge_labels_1 = nx.get_edge_attributes(G,'weight')
nx.draw(G,nx.shell_layout(G),with_labels = True)
nx.draw_networkx_edge_labels(G,nx.shell_layout(G),edge_labels=edge_labels_1)