import numpy as np
import networkx as nx
def perform_sweep(all_graphs,n_angle):

  #Number of nodes in the graph
  num_nodes = nx.number_of_nodes(all_graphs[0])
  
  n_quad = 4
  predecessors = dict.fromkeys(range(n_quad))
  successors = dict.fromkeys(range(n_quad))
  starting_nodes = dict.fromkeys(range(n_quad))
  
  for q in range(0,n_quad):
    
    #Getting the graph for this quadrant
    current_graph = all_graphs[q]

    quad_pred = dict.fromkeys(range(num_nodes))
    quad_suc = dict.fromkeys(range(num_nodes))
    
    starting_nodes[q] = [x for  x in current_graph.nodes() if current_graph.in_degree(x) == 0]
    
    for n in range (0, num_nodes):
      #The predecessors for this quadrant's graph
      quad_pred[n] = current_graph.predecessors(n)
      #The successors for this quadrants's graph
      quad_suc[n] = current_graph.successors(n)
    
    predecessors[q] = quad_pred
    successors[q] = quad_suc
  
  #All predecessors and successors have been populated for all nodes for all quadrant configurations of the graphs.
  #Total Number of tasks in this problem.
  n_tasks = num_nodes*n_angle*n_quad
  num_stages = 1
  
  #Initially, the current_nodes are the starting nodes.
  current_nodes = starting_nodes
  
  while n_tasks > 0:
    
    potentially_next = dict.fromkeys(range(4))
    #We get nodes that are potentially next per quadrant and remove the current nodes from the quadrant.
    for q in range(0,n_quad):
      #Successors for this quadrant.
      quad_suc = successors[q]
      #Predecessors of this quadrant.
      quad_pred = predecessors[q]
      #Predecessors to remove for this quadrant.
      nodes_to_remove = current_nodes[q]
      
      for n in range(0,len(nodes_to_remove)):
        node_to_remove = nodes_to_remove[n]
        
      
      #Updating global predecessors dictionary.
      predecessors[q] = quad_pred
        
        
      
    #if (num_stages != 1):
      #We check if the predecessors have been solved for this angle.