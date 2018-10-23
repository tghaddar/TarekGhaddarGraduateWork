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
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sweep_solver import get_fastest_path
from sweep_solver import get_heaviest_path
from sweep_solver import get_weight_sum
plt.close("all")

G = nx.DiGraph()
G2 = nx.DiGraph()

num_nodes = 5

for n in range(0,num_nodes):
  G.add_node(n)
  G2.add_node(n)

G.add_edge(0,1,weight = 3)
G.add_edge(1,2,weight = 10)
G.add_edge(2,3,weight = 2)
G.add_edge(3,4,weight = 8)
#plt.figure("Graph 1")
#nx.draw(G,nx.shell_layout(G),with_labels = True)

G2.add_edge(4,3,weight = 1)
G2.add_edge(3,2,weight = 8)
G2.add_edge(2,1,weight = 2)
G2.add_edge(1,0,weight = 10)

#plt.figure("Graph 2")
#nx.draw(G2,nx.shell_layout(G2),with_labels = True)

path1 = nx.all_simple_paths(G,0,4)
path2 = nx.all_simple_paths(G2,4,0)



graphs = [G,G2]
paths = [path1,path2]
#Getting the heaviest path in each graph.
for p in range(0,len(paths)):
  current_path = paths[p]
  heavy_path = get_heaviest_path(graphs[p],current_path)
  paths[p] = heavy_path


for n in range(0,num_nodes):
  fastest_path,weight_sum = get_fastest_path(graphs,paths,n)
    
  primary_graph = graphs[fastest_path]
  primary_path = paths[fastest_path]
  primary_index = primary_path.index(n)
  
  #Looping through remaining path to add potential delays for this node.
  for p in range(0,len(paths)):
    secondary_path = paths[p]
    secondary_graph = graphs[p]
    if p == fastest_path:
      continue
    #Check if this node exists in the secondary path.
    secondary_index = -1
    try:
      secondary_index = secondary_path.index(n)
    except:
      continue
    
    weight_sum_secondary = get_weight_sum(graphs[p],paths[p],n)
    delay = weight_sum_secondary - weight_sum
    time_to_solve = primary_graph[n][primary_path[primary_index+1]]['weight']
    delay = time_to_solve - delay
    if (delay > 0):
      #Add this delay to the current node's solve time in the secondary graph.
      next_node = secondary_path[secondary_index+1]
      secondary_graph[n][next_node]['weight'] += delay

for line in nx.generate_edgelist(G,data=True):
  print(line)

print("\n")
print("G2\n")
for line in nx.generate_edgelist(G2,data=True):
  print(line)