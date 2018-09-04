import numpy as np
import warnings
import networkx as nx
warnings.filterwarnings("ignore", category=DeprecationWarning)

#This function computes the solve time for a sweep for each octant. 
#Params:
#A list of TDGs.
#The number of cells per subset.
#The grind time (time to solve an unknown on a machine)
#The communication time per byte. 
t_byte = 1e-09
#The number of bytes to communicate per subset.
#The message latency time.
m_l = 1

def compute_solve_time(tdgs,t_byte,m_l,cells_per_subset):
  time = 0
  #Number of nodes in the graph.
  num_nodes = nx.number_of_nodes(tdgs[0])
  
  #Looping over the graphs.
  for ig in range(0,len(tdgs)):
    #The current graph
    graph = tdgs[ig]
    
    #Getting the starting node for this graph.
    starting_node = [x for x in graph.nodes() if graph.in_degree(x)==0]
    
    #Storing the successors for each node in the graph.
    successors = dict.fromkeys(range(num_nodes))
    predecessors = dict.fromkeys(range(num_nodes))
    for n in range(0,num_nodes):
      successors[n] = list(graph.successors(n))
      predecessors[n] = list(graph.predecessors(n))
      
  
  return time
    