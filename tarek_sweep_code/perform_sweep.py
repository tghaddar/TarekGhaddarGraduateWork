import numpy as np
import networkx as nx
import collections
from itertools import chain
from resolve_conflict import resolve_conflict
from copy import copy
def perform_sweep(all_graphs,n_angle):

  #Number of nodes in the graph
  num_nodes = nx.number_of_nodes(all_graphs[0])
  #Dictionary storing the current nodes at each stage.
  wave = {}
  
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
      quad_pred[n] = list(current_graph.predecessors(n))
      #The successors for this quadrants's graph
      quad_suc[n] = list(current_graph.successors(n))
    
    predecessors[q] = quad_pred
    successors[q] = quad_suc
  
  #All predecessors and successors have been populated for all nodes for all quadrant configurations of the graphs.
  #Total Number of tasks in this problem.
  n_tasks = num_nodes*n_angle*n_quad
  num_stages = 0
  
  
  
  #Initially, the current_nodes are the starting nodes.
  current_nodes = starting_nodes
  
  counter = 1
  while n_tasks > 0:
    #We know the current nodes from the previous stages successors. 
    #However, we need to make sure that the nodes up to be solved do not have any predecessors. If they do, they are not ready to be solved.
    if (counter > 1):
      #We get nodes that are potentially next per quadrant and remove the current nodes from the quadrant.
      for q in range(0,n_quad):
        #Predecessors of this quadrant.
        quad_pred = predecessors[q]
        #Predecessors to remove for this quadrant.
        potentially_remove = current_nodes[q]
        #len_potentially_remove = len(potentially_remove)
        n = 0
        while n < len(potentially_remove):
          #The node we are checking to make sure it has no predecessors.
          node_to_remove = potentially_remove[n]
          #If it still has predecessors, we remove it.
          if quad_pred[node_to_remove]:
            del potentially_remove[n]
          else:
            n+=1 
            
        current_nodes[q] = potentially_remove
    
    print(current_nodes)
    #At this point, for every quadrant, we have removed any nodes from current_nodes that is not ready to be solved, meaning it still has predecessors. If a predecessors list for a node is empty, then we know it should be solved and belongs in current_nodes.
    
    #Checking for conflicts. We know our current nodes being solved in each quadrant. We look for any duplicate nodes in current_nodes, across quadrants.
    #All the current nodes,across all quadrants flattened into a 1d array.
    all_current_nodes = np.array(list(chain.from_iterable(current_nodes.values())))
    #The nodes that have a conflict.
    conflicted_nodes = [item for item, count in collections.Counter(all_current_nodes).items() if count > 1]
    current_nodes_with_conflicts = copy(current_nodes)
    #print("Conflicted: ", current_nodes_with_conflicts)
    #Checking 
    for  n in range(0,len(conflicted_nodes)):
      current_node = conflicted_nodes[n]
      #Pulling quadrants where this node exists.
      quadrants = [k for k in current_nodes for x in current_nodes[k] if x == current_node]
      #Getting the graphs that we need.
      conflict_graphs = [all_graphs[i] for i in quadrants]
      winning_quadrant = resolve_conflict(conflict_graphs,quadrants,current_node,num_nodes)
      #The quadrant with the winning node keeps it in it's current nodes. The other quadrants remove this node.
      #This is the current_nodes for the winning quadrant.
      current_nodes_wq = current_nodes[winning_quadrant]
      
      current_nodes = {k:[x for x in current_nodes[k] if x!=current_node] for k in current_nodes if k!=winning_quadrant}
      #Adding back the winning quadrant 
      current_nodes[winning_quadrant] = current_nodes_wq
    
    
    print("Pre Conflict: ", current_nodes_with_conflicts)
    print("Post Conflict: ", current_nodes)
    #We have successfully distilled the current_nodes to things that will only be solved at this stage.
    #We now have to remove all these nodes from the dictionary of predecessors FOR RESPECTIVE QUADRANTS.
    for q in range(0,n_quad):
      #There should only be one current node for each quadrant at this point.
      quad_current_nodes = current_nodes[q]
      #If there are current nodes for this stage.
      if (quad_current_nodes):
        quad_pred = predecessors[q]
         #Removing the current nodes as a predecessor for any other nodes in this quadrant. It has been solved.
        for n in range(0,len(quad_current_nodes)):
          quad_pred = {k:[x for x in quad_pred[k] if x!=quad_current_nodes[n]] for k in quad_pred}
        predecessors[q] = quad_pred
      

    
    current_nodes_copy = copy(current_nodes)
    #Add the current nodes to this stage.
    wave[num_stages] = current_nodes_copy
    test = copy(wave[num_stages])
    test = copy({k:[x for x in test[k]] for k in test if test[k]})
    if not test:
      break
      
    num_stages += 1
    #Subtracting the current nodes from the task
    n_tasks -= sum(map(len,current_nodes.values()))
    counter += 1
    #We now need to set the successors for the next stage. We also have to add back any current nodes that were removed during the conflict.
    for q in range(0,n_quad):
      quad_suc = successors[q]
      quad_current_nodes = current_nodes[q]
      conflict_nodes = current_nodes_with_conflicts[q]
      #The nodes we need to add back.
      diff = list(set(conflict_nodes) - set(quad_current_nodes))
      if (diff):
        if diff[0] < 0:
          diff = []
      #NEED TO FLATTEN THIS LIST. But the logic is good.
      full_list = [quad_suc[x] for x in quad_current_nodes]
      flat_list = [item for sublist in full_list for item in sublist]
      current_nodes[q] = list(set(flat_list).union(set(diff)))
  
      
  return num_stages,wave
      
    
    
    
    