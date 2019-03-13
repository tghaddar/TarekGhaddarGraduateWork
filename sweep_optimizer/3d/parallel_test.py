# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 18:03:31 2019

@author: tghad
"""

from sweep_solver import get_heaviest_path_faster
import multiprocessing as mp
#from numba import autojit,prange
import numpy as np
from build_adjacency_matrix import build_graphs
import time
from copy import deepcopy
from concurrent import futures





def make_edges_universal(graphs):
  
  num_nodes = graphs[0].number_of_nodes()-1
  num_graphs = len(graphs)
  
  #Looping over all graphs.
  for g in range(0,num_graphs):
    #The current_graph which we will alter.
    graph = graphs[g]
    
    #Getting the starting node of this graph.
    start_node = [x for x in graph.nodes() if graph.in_degree(x) == 0][0]
    #A list storing the heaviest path length to each node.
    heavy_path_lengths = [0.0]*num_nodes
    #Looping over nodes to get the longest path to each node.
    for n in range(0,num_nodes):
      #If the starting node is the target node we know that path length is zero.
      if (start_node == n):
        continue
      
      #Gets the heaviest path length and its weight. 
      heaviest_path_length = get_heaviest_path_faster(graph,start_node,n)
      #Storing this value in heavy_path_lengths.
      heavy_path_lengths[n] = heaviest_path_length
      
    #Storing the heavy path lengths as the weight value to all preceding edges.
    for n in range(0,num_nodes):
      
      #The starting node has no preceding edges so we skip it.
      if (n != start_node):
        #Getting the weight we want for preceding edges.
        new_weight = heavy_path_lengths[n]
        #Getting the incoming edges to this node.
        incoming_edges = list(graph.in_edges(n,'weight'))
        for edge in incoming_edges:
          graph[edge[0]][edge[1]]['weight'] = new_weight

    #Adding the value of the last edge (end_node to the dummy -1 node).
    true_end_node = list(graph.predecessors(-1))[0]
    pred_end_node = list(graph.predecessors(true_end_node))[0]
    graph[true_end_node][-1]['weight'] += graph[pred_end_node][true_end_node]['weight']
    
    graphs[g] = graph
  return graphs


numrow = 50
numcol = 50

adjacency_matrix = np.genfromtxt('matrices_49.csv',delimiter=",")


graphs = build_graphs(adjacency_matrix,numrow,numcol)
num_graphs = len(graphs)

#Adding the uniform edge labels.
for g in range(0,num_graphs):
  graph = graphs[g]
  graph.add_weighted_edges_from((u,v,1) for u,v in graph.edges())

G,G1,G2,G3 = graphs
G_copy = deepcopy(G)
num_nodes = G.number_of_nodes()-1
start_node = [x for x in G.nodes() if G.in_degree(x) == 0][0]


#@autojit
#def inner_loop():
#  for n in prange(0,num_nodes):
#    if (start_node != n):
#      heavy_path_lengths_par[n] = get_heaviest_path_faster(G,start_node,n)
#
#inner_loop()

def inner_loop(n):
  if start_node == n:
    return 0.0
  
  return get_heaviest_path_faster(G,start_node,n)

heavy_path_lengths_par = []
start_par = time.time() 
all_nodes = range(0,num_nodes)

print("Num cores: ", mp.cpu_count())

with mp.Pool(4) as p:
  heavy_path_lengths_par = p.map(inner_loop,all_nodes)
end_par = time.time()


 #A list storing the heaviest path length to each node.
heavy_path_lengths = [0.0]*num_nodes
start_serial = time.time()
#Looping over nodes to get the longest path to each node.
for n in range(0,num_nodes):
  #If the starting node is the target node we know that path length is zero.
  if (start_node == n):
    continue
  
  #Gets the heaviest path length and its weight. 
  heaviest_path_length = get_heaviest_path_faster(G_copy,start_node,n)
  #Storing this value in heavy_path_lengths.
  heavy_path_lengths[n] = heaviest_path_length

end_serial = time.time()

if heavy_path_lengths_par  == heavy_path_lengths:
  print("Success")
else:
  print("Failure")

print("Parallel Time: ", end_par - start_par)
print("Serial Time: ", end_serial - start_serial)


