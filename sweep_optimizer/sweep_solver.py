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

def compute_solve_time(tdgs):
  #Number of nodes in the graph.
  num_nodes = nx.number_of_nodes(tdgs[0])
  for ig in range(0,len(tdgs)):
    
    #The current graph
    graph = tdgs[ig]
    
    