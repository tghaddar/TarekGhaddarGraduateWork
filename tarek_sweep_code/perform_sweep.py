import numpy as np
import networkx as nx
def perform_sweep(all_graphs,n_angle):

  num_subsets = nx.number_of_nodes(all_graphs[0])
  
  n_quad = 4
  predecessors = {}
  successors = {}
  
  for q in range(0,n_quad):
    
    #Getting the graph for this quadrant
    current_graph = all_graphs[q]
    #Number of nodes in the graph.
    num_nodes = current_graph.number_of_nodes()
    quad_pred = {}
    quad_suc = {}
    #starting_node = [x for  x in current_graph.nodes() if current_graph.in_degree(x) == 0][0]
    for n in range (0, num_nodes):
      #The predecessors for this quadrant's graph
      quad_pred[n] = current_graph.predecessors(n)
      #The successors for this quadrants's graph
      quad_suc[n] = current_graph.successors(n)
    
    predecessors[q] = quad_pred
    successors[q] = quad_suc
  
  #All predecessors and successors have been populated for all nodes for all quadrant configurations of the graphs.
  #Total Number of tasks in this problem.
  n_tasks = num_subsets*n_angle*n_quad
  num_stages = 0
  
    
  
  