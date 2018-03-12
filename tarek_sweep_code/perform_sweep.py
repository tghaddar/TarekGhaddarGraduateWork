import numpy as np
import networkx as nx
import collections
from itertools import chain
from resolve_conflict import resolve_conflict
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
    
    #We get nodes that are potentially next per quadrant and remove the current nodes from the quadrant.
    for q in range(0,n_quad):
      #Successors for this quadrant.
      quad_suc = successors[q]
      #Predecessors of this quadrant.
      quad_pred = predecessors[q]
      #Predecessors to remove for this quadrant.
      potentially_remove = current_nodes[q]
      
      for n in range(0,len(potentially_remove)):
        node_to_remove = potentially_remove[n]
        
        #Check if this node has no predecessors.
        if not quad_pred[node_to_remove]:
          #If it doesn't have any predecessors, then it is ready to be solved.
          pass
        else:
          #If it still has predecessors, it is not ready to be solved and should not be in current nodes.
          potentially_remove = [x for x in potentially_remove if x!= node_to_remove]
        
      
      current_nodes[q] = potentially_remove
        
    #At this point, for every quadrant, we have removed any nodes from current_nodes that is not ready to be solved, meaning it still has predecessors. If a predecessors list for a node is empty, then we know it should be solved and belongs in current_nodes.
    
    #Checking for conflicts. We know our current nodes being solved in each quadrant. We look for any duplicate nodes in current_nodes, across quadrants.
    #All the current nodes,across all quadrants flattened into a 1d array.
    all_current_nodes = np.array(list(chain.from_iterable(current_nodes.values())))
    #The nodes that have a conflict.
    conflicted_nodes = [item for item, count in collections.Counter(all_current_nodes).items() if count > 1]
    #Checking 
    for  n in range(0,len(conflicted_nodes)):
      current_node = conflicted_nodes[n]
      #Pulling quadrants where this node exists.
      quadrants = [k for k in current_nodes for x in current_nodes[k] if x == current_node]
      #Getting the graphs that we need.
      conflict_graphs = [all_graphs[i] for i in quadrants]
      winning_quadrant = resolve_conflict(conflict_graphs,quadrants,current_node)
      #The quadrant with the winning node keeps it in it's current nodes. The other quadrants remove this node.
    
    #We have to remove it entirely from the dictionary of predecessors.
    quad_pred = {k:[x for x in quad_pred[k] if x!= node_to_remove] for k in quad_pred}
        
    
    #Updating global predecessors dictionary.
    predecessors[q] = quad_pred
    break
      
    #if (num_stages != 1):
      #We check if the predecessors have been solved for this angle.