#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 13:38:39 2018
Unit test for conflict corrector.
@author: tghaddar
"""
import networkx as nx
import matplotlib.pyplot as plt
import warnings

plt.close("all")

G = nx.DiGraph()
G2 = nx.DiGraph()

num_nodes = 5

for n in range(0,5):
  G.add_node(n)
  G2.add_node(n)

G.add_edge(0,1,weight = 3)
G.add_edge(1,2,weight = 10)
G.add_edge(2,3,weight = 2)
G.add_edge(3,4,weight = 8)
plt.figure("Graph 1")
nx.draw(G,nx.shell_layout(G),with_labels = True)

G2.add_edge(4,3,weight = 1)
G2.add_edge(3,2,weight = 8)
G2.add_edge(2,1,weight = 2)
G2.add_edge(1,0,weight = 10)

plt.figure("Graph 2")
nx.draw(G2,nx.shell_layout(G2),with_labels = True)

path1 = nx.all_simple_paths(G,0,4)
path2 = nx.all_simple_paths(G2,4,0)

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
  
  #Looping through other paths for this node.
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
    
    #If the primary path is faster than the secondary path to reach this node, then we set faster_path to True.
    faster_path = True
    if weight_sum_path < weight_sum:
      faster_path = False
    
    if (faster_path):
      #How much faster it starts solving than the secondary path. 
      delay = weight_sum_path - weight_sum
      #Get the time to solve this node.
      time_to_solve = G[current_node][primary_path[i+1]]['weight']
      delay = time_to_solve - delay
      #Check if delay is positive, then we need to add weight to the secondary graph.
      if (delay > 0):
        #We add this delay to the node 1's solve time in the secondary graph.
        next_node = path[node_position+1]
        G2[current_node][next_node]['weight'] += delay
    else:
      delay = weight_sum - weight_sum_path
      time_to_solve = G2[current_node][path[node_position+1]]['weight']
      delay = time_to_solve - delay
      if (delay > 0):
        next_node = primary_path[i+1]
        G[current_node][next_node]['weight'] += delay
      



for line in nx.generate_edgelist(G,data=True):
  print(line)

print("\n")
print("G2\n")
for line in nx.generate_edgelist(G2,data=True):
  print(line)