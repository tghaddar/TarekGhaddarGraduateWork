import networkx as nx
#We need to determine which quadrant wins for this node. We'll do this with seeing who has a longer depth of graph (more work left). 
def resolve_conflict(graphs,quadrants,node,num_nodes):
  
  n_quad = len(quadrants)
  path_lengths = []
  for q in range(0,n_quad):
    #The current graph
    G = graphs[q]
    #Last node.
    last_node = list(G.nodes())[num_nodes-1]
    try:
      path_length = nx.shortest_path_length(G,node,last_node)
    except nx.NetworkXNoPath:
      path_length = 0
    
    path_lengths.append(path_length)
  
  
  #Finding the max path length.
  max_length = max(path_lengths)
  #Getting the indices that contain this maximum path length.
  max_indices = [i for i, j in enumerate(path_lengths) if j == max_length]
  
  #If there's only one index that has this, then that is the winning quadrant.
  if len(max_indices) == 1:
    winning_quadrant = quadrants[max_indices[0]]
  #Otherwise we have to break the tie.
  else:
    tied_quadrants = [quadrants[i] for i in max_indices]
    winning_quadrant = min(tied_quadrants)
  
  return winning_quadrant
  