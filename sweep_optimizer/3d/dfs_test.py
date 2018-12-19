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
#Getting successors to the starting node.
successors = list(copy_graph.successors(start_node))
s = 0
count = 0
while s < len(successors):
  #Make the list of successors unique each time.
  successors = list(set(successors))
  #Current successor.
  succ = successors[s]
  
  #Check the predecessors of this successor.
  predecessors = list(copy_graph.predecessors(succ))
  num_preds = len(predecessors)
  
  #If there is only one predecessor, we remove this from our list of successors, as there is nothing to evaluate, and add it's successors to the list.
  if num_preds == 1:
    successors.remove(succ)
    try:
      successors += list(copy_graph.successors(succ))
    except:
      continue
  #Otherwise we check who has the largest edge weight and set those to that.
  else:
    max_weight = 0
    for p in range(0,num_preds):
      pred = predecessors[p]
      edge_weight = G[pred][succ]['weight']
      if (edge_weight > max_weight):
        max_weight = edge_weight
    
    #Now that we have the max weight, we set that as the new edge weight for all predecessors.
    for p in range(0,num_preds):
      pred = predecessors[p]
      G[pred][succ]['weight'] = max_weight
    
    #We remove this from our list of successors.
    successors.remove(succ)
    #We add in the successors of this node.
    try:
      successors += list(copy_graph.successors(succ))
    except:
      continue
  count += 1


plt.figure("Graph 0 Post Universal ish Time")
edge_labels_1 = nx.get_edge_attributes(G,'weight')
nx.draw(G,nx.shell_layout(G),with_labels = True)
nx.draw_networkx_edge_labels(G,nx.shell_layout(G),edge_labels=edge_labels_1)