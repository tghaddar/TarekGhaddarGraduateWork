#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 12:03:31 2018
#Unit test for the graph depth traversal.
@author: tghaddar
"""

import networkx as nx

def sum_weights_of_path(graph,path):
  weight_sum = 0.0
  for n in range(0,len(path)-1):
    node1 = path[n]
    node2 = path[n+1]
    weight = graph[node1][node2]['weight']
    print(weight)
    weight_sum += graph[node1][node2]['weight']
    
  
  return weight_sum
    

G = nx.DiGraph()

for n in range(0,8):
  G.add_node(n)

G.add_edge(0,1,weight = 1)
G.add_edge(0,2,weight = 1)

G.add_edge(1,3,weight = 2)
G.add_edge(2,3,weight = 99)

G.add_edge(1,5,weight = 2)
G.add_edge(5,6,weight = 198)
G.add_edge(6,7,weight = 1)

G.add_edge(3,4,weight = 3)
G.add_edge(4,7,weight = 1)

for line in nx.generate_edgelist(G,data=True):
  print(line)
  
  
paths = nx.all_simple_paths(G,0,7)

max_weight = 0.0
for path in paths:
  weight = sum_weights_of_path(G,path)
  if weight > max_weight:
    max_weight = weight
  
print(max_weight)