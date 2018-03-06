import networkx as nx
import numpy as np
def perform_sweep(all_graphs,n_angle):

  num_subsets = nx.number_of_nodes(all_graphs[0])
  
  n_quad = 4
  predecessors = np.empty(n_quad)
  successors = np.empty(n_quad)
  
  for q in range(0,n_quad):
    
    #Getting the graph for this quadrant
    current_graph = all_graphs[q]
    #Number of nodes in the graph.
    num_nodes = current_graph.number_of_nodes()
    
    starting_node = [x for  x in current_graph.nodes() if current_graph.in_degree(x) == 0][0]
    
    #The predecessors for this quadrant's graph
    quad_pred = nx.predecessor(current_graph,starting_node)
    print(quad_pred)
    #The successors for this quadrants's graph
    quad_suc = nx.predecessor(current_graph,starting_node)
    
#    predecessors[q] = quad_pred
#    successors[q] = quad_suc
