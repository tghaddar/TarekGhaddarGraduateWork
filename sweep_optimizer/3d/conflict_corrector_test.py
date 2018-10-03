#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 13:38:39 2018
Unit test for conflict corrector.
@author: tghaddar
"""
import networkx as nx
import matplotlib.pyplot as plt

#plt.close("all")

G = nx.DiGraph()
G2 = nx.DiGraph()

num_nodes = 3

for n in range(0,num_nodes):
  G.add_node(n)
  G2.add_node(n)

G.add_edge(0,1,weight = 3)
G.add_edge(1,2,weight = 10)
#plt.figure("Graph 1")
#nx.draw(G,nx.shell_layout(G),with_labels = True)


G2.add_edge(2,1,weight = 5)
G2.add_edge(1,0,weight = 8)
#plt.figure("Graph 2")
#nx.draw(G2,nx.shell_layout(G2),with_labels = True)

path1 = nx.all_simple_paths(G,0,2)
path2 = nx.all_simple_paths(G2,2,0)

paths = []
for path in path1:
  paths.append(path)

for path in path2:
  paths.append(path)
  
primary_path = paths[0]
num_edges = len(primary_path) - 1

for i in range(1,num_nodes-1):
  
  current_node = primary_path[i]
  
  #Getting the sum of weights up to this point.
  weight_sum = 0.0
  for j in range(0,i):
    node1 = primary_path[j]
    node2 = primary_path[j+1]
    weight_sum += G[node1][node2]['weight']
  
  
  for p in range(1,len(paths)):
    
    path = paths[p]
    #Finding the index of the current primary node in this path.
    node_position = path.index(current_node)
    
    weight_sum_path = 0.0
    #Getting the sum of the wieghts up to this node.
    for j in range(0,node_position):
      node1 = path[j]
      node2 = path[j+1]
      weight_sum_path += G2[node1][node2]['weight']
    
    print(weight_sum_path)