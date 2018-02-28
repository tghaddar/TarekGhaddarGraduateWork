import networkx as nx
import numpy as np

def build_graph(adjacency_list):
  
  num_subsets = len(adjacency_list)
  
  #The true adjacency matrix
  adjacency_matrix = np.zeros((num_subsets,num_subsets))
  
  for i in range(0,num_subsets):
    
    neighbors = adjacency_list[i]
    for j in range(0,len(neighbors)):
      
      adjacency_matrix[i][neighbors[j]] = 1
      
  
  #Now that we have a true adjacency matrix, we can build the graph.
  G = nx.DiGraph(adjacency_matrix)
  return G
  