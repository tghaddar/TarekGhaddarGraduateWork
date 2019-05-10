import numpy as np
import warnings
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import copy
from build_adjacency_matrix import build_graphs
from build_adjacency_matrix import build_adjacency
from copy import deepcopy
from utilities import get_ijk
from utilities import get_ij,get_ss_id
from math import isclose
from matplotlib.pyplot import imshow,pause
from mesh_processor import get_cells_per_subset_2d
import time
import operator
warnings.filterwarnings("ignore", category=DeprecationWarning)

#This function computes the solve time for a sweep for each octant. 
#Params:
#A list of TDGs.
#The number of cells per subset.
#The grind time (time to solve an unknown on a machine)
#The communication time per byte. 


#Plots graphs at a specific time. Will help with debugging.
def plot_graphs(graphs,t):
  
  num_graphs = len(graphs)
  grange = range(0,num_graphs)
  for g in grange:
    plt.figure(str(g)+str(t))
    plt.title("Time " + str(t) + " Graph " + str(g))
    edge_labels_1 = nx.get_edge_attributes(graphs[g],'weight')
    if g < 3:
      nx.draw(graphs[g],nx.spectral_layout(graphs[g],weight='weight'),with_labels = True)
      nx.draw_networkx_edge_labels(graphs[g],nx.spectral_layout(graphs[g],weight='weight'),edge_labels=edge_labels_1,font_size=5)
    else:
      nx.draw(graphs[g],nx.spectral_layout(graphs[g],weight='weight'),with_labels = True)
      nx.draw_networkx_edge_labels(graphs[g],nx.spectral_layout(graphs[g],weight='weight'),edge_labels=edge_labels_1,font_size=5)
    plt.savefig("debug_graph_plots/graph_"+str(t)+"_"+str(g)+".pdf")
    plt.close()
    

#A modified version of networkx simple paths algorithm. This behaves as a weight based depth traversal algorithm.
def all_simple_paths_modified(G, source, target, time_to_solve_graph, cutoff=None):

    if source not in G:
        raise nx.NodeNotFound('source node %s not in graph' % source)
    if target not in G:
        raise nx.NodeNotFound('target node %s not in graph' % target)
    if source == target:
        return []
    if cutoff is None:
        cutoff = len(G) - 1
    return _all_simple_paths_graph_modified(G, source, target,time_to_solve_graph, cutoff=cutoff)


def _all_simple_paths_graph_modified(G, source, target, time_to_solve_graph, cutoff=None):

    visited = [source]
    start_time_source = 0.0
    try:
      start_time_source = list(G.in_edges(source,'weight'))[0][2]
    except:
      start_time_source = 0.0
    stack = [iter(G[source])]
    if cutoff == start_time_source:
      yield source
    elif start_time_source > cutoff:
      return 
    while stack:
        children = stack[-1]
        child = next(children, None)
        start_time = list(G.in_edges(child,'weight'))[0][2]
        if child is None:
          stack.pop()
          visited.pop()
        elif child == target:
          stack.pop()
          visited.pop()
        elif start_time == cutoff:
          yield child
        elif start_time < cutoff:
          #Making sure that it's actually solving and not just waiting to communicate.
          if start_time + time_to_solve_graph[child] > cutoff:
            yield child
        elif start_time > cutoff:
          #Checking to see the source is actually solving and not just waiting to communicate.
          if start_time_source + time_to_solve_graph[source] > cutoff:
            yield source

def modify_downstream_edges(G,source,target,modified_edges,delay):
  
  visited = [source]
  stack = [iter(G[source])]

  while stack:
    children = stack[-1]
    child = next(children,None)
    if child is None:
      stack.pop()
      visited.pop()
    else:
      prev_node = visited[-1]
      edge = (prev_node,child)
      if not modified_edges:
        G[prev_node][child]['weight'] += delay
        modified_edges.append(edge)
      elif (edge not in modified_edges):
        G[prev_node][child]['weight'] += delay
        modified_edges.append(edge)
      if child not in visited:
        visited.append(child)
        stack.append(iter(G[child]))
      
  return G

def modify_downstream_edges_faster(G,graph_index,source,modified_edges,time_to_solve,og_delay):
#  downstream_nodes = nx.descendants(G,source)
#  for node in downstream_nodes:
#  
#    #Getting incoming edges to this node.
#    in_edges = G.in_edges(node)
#    #Get the weights of  in_edges.
#    weights = [z for x,y,z in in_edges]
#    #The maximum weight (which is when this downstream node is ready to solve)
#    ready_to_solve = copy(max(weights))
#    
#    for u,v in in_edges:
#      if (u == source or u in downstream_nodes):
#        if not modified_edges:
#          G[u][v]['weight'] += delay
#          modified_edges.append((u,v))
#        elif (u,v) not in modified_edges:
#          G[u][v]['weight'] += delay
#          modified_edges.append((u,v))
  
  
  downstream_nodes = list(nx.descendants(G,source))
  #Add the source node to the downstream nodes.
  downstream_nodes = [source] + downstream_nodes
  num_downstream_nodes = len(downstream_nodes)
  #We get when each downstream node is ready to solve.
  ready_to_solve_all = {}
  for n in range(0,num_downstream_nodes):
    current_node = downstream_nodes[n]
    #Get incoming edge with the maximum weight to this node.
    ready_to_solve_all[current_node] = get_max_incoming_weight(G,current_node)
  
  
  #Sorting the downstream nodes in order of when they solve.
  ready_to_solve_all = dict(sorted(ready_to_solve_all.items(),key=lambda x:x[1]))
  
  
  for k,val in ready_to_solve_all.items():
    #The current node.
    node = k
    #When the current node is ready to solve.
    ready_to_solve = val
    #Get outgoing edges of this node.
    out_edges = G.out_edges(node)
    for u,v in out_edges:
      if (v in downstream_nodes):
        if not modified_edges:
          delay = time_to_solve[graph_index][node] + ready_to_solve - G[u][v]['weight']
          if delay > 0.0:
            G[u][v]['weight'] += delay
            ready_to_solve_all[v] = get_max_incoming_weight(G,v)
            #modified_edges.append((u,v))
        elif (u,v) not in modified_edges:
          delay = time_to_solve[graph_index][node] + ready_to_solve - G[u][v]['weight']
          if delay > 0.0:
            G[u][v]['weight'] += delay
            ready_to_solve_all[v] = get_max_incoming_weight(G,v)
            #modified_edges.append((u,v))
  
  return G

def get_max_incoming_weight(G,node):
  
  in_edges = list(G.in_edges(node,'weight'))
  weights = [z for x,y,z in in_edges]
  
  return max(weights)

def get_subset_cell_dist(num_total_cells,global_subset_boundaries):
  
  #Approximately two cells per mini subset.
  num_mini_sub = num_total_cells/2.0
  num_subsets = len(global_subset_boundaries)
  first_sub = global_subset_boundaries[0]
  last_sub = global_subset_boundaries[num_subsets-1]
  global_x_min = first_sub[0]
  global_y_min = first_sub[2]
  global_z_min = first_sub[4]
  
  global_x_max = last_sub[1]
  global_y_max = last_sub[3]
  global_z_max = last_sub[5]
  
  #global x length
  a = global_x_max - global_x_min
  #global y length
  b = global_y_max - global_y_min
  #global z length
  c = global_z_max - global_z_min
  #Number of mini subs in x
  nsx = a*pow((num_mini_sub/(a*b*c)),1/3)
  #Number of mini subs in y
  nsy = b*pow((num_mini_sub/(a*b*c)),1/3)
  #Number of mini subs in z
  nsz = c*pow((num_mini_sub/(a*b*c)),1/3)
  cells_per_subset = []
  for s in range(0,num_subsets):
    #The boundaries for this node.
    bounds = global_subset_boundaries[s]
    
    #Ratio of x-length to y-length of the subset.
    x_ratio = (bounds[1]-bounds[0])/(global_x_max - global_x_min)
    y_ratio = (bounds[3]-bounds[2])/(global_y_max - global_y_min)
    z_ratio = (bounds[5]-bounds[4])/(global_z_max - global_z_min)
    #Approx number of mini subsets in each direction.
    num_sub_y = nsy*y_ratio
    num_sub_x = nsx*x_ratio
    num_sub_z = nsz*z_ratio
    cells_in_subset = 2.0*num_sub_y*num_sub_x*num_sub_z
    cells_per_subset.append(cells_in_subset)
    
  return cells_per_subset

def plot_subset_boundaries(global_3d_subset_boundaries,num_subsets):
  fig = plt.figure(1)
  ax = fig.gca(projection='3d')
  subset_centers = []
  layer_colors = ['b','r']
  layer = 0
  for i in range(0,num_subsets):
  
    subset_boundary = global_3d_subset_boundaries[i]
    xmin = subset_boundary[0]
    xmax = subset_boundary[1]
    ymin = subset_boundary[2]
    ymax = subset_boundary[3]
    zmin = subset_boundary[4]
    zmax = subset_boundary[5]
    if (zmax == 10.0):
      layer = 1
    else:
      layer = 0
  
    center_x = (xmin+xmax)/2
    center_y = (ymin+ymax)/2
    center_z = (zmin+zmax)/2
  
    subset_centers.append([center_x, center_y, center_z])
  
    x = [xmin, xmax, xmax, xmin, xmin,xmax,xmax,xmin,xmin,xmin,xmin,xmin,xmax,xmax,xmin,xmin]
    y = [ymin, ymin, ymax, ymax, ymin,ymin,ymin,ymin,ymin,ymax,ymax,ymin,ymin,ymax,ymax,ymin]
    z = [zmin, zmin, zmin, zmin, zmin,zmin,zmax,zmax,zmin,zmin,zmax,zmax,zmax,zmax,zmax,zmax]
  
    ax.plot(x,y,z,layer_colors[layer])
    
    x2 = [xmax,xmax]
    y2 = [ymax,ymax]
    z2 = [zmax,zmin]
    ax.plot(x2,y2,z2,layer_colors[layer])
  
  plt.savefig("subset_plot.pdf")
  
#Checking if the current node shares x or y subset boundaries.
def find_shared_bound(node,succ,num_row,num_col,num_plane):
  bounds_check = []
  
  i_node,j_node,k_node = get_ijk(node,num_row,num_col,num_plane)
  i_succ,j_succ,k_succ = get_ijk(succ,num_row,num_col,num_plane)
  
  #If they are in different layers, we know the shared boundary is the xy plane.
  if (k_node != k_succ):
    bounds_check = 'xy'
  #If the two subsets are in the same layer.
  else:
    if (i_node == i_succ):
      bounds_check = 'xz'
    else:
      bounds_check = 'yz'
    
  return bounds_check

def add_edge_cost(graphs,global_subset_boundaries,cells_per_subset, bdy_cells_per_subset,machine_params,num_row,num_col):
    
  num_graphs = len(graphs)
  num_subsets = len(global_subset_boundaries)
  #Storing the time to solve and communicate each subset.
  time_to_solve = [[None]*num_subsets for g in range(num_graphs)]
  t_u,upc,upbc,t_comm,latency,m_l = machine_params
  
  for ig in range(0,num_graphs):
    graph = graphs[ig]
    for e in graph.edges():
      #The starting node of this edge.
      node = e[0]
      #Checking the successors.
      node_successors = list(graph.successors(node))
      bounds = [False, False]
      #Determining if this subset has neighbors in both x and y.
      for n in range(0,len(node_successors)):
        neighbor = node_successors[n]
        if neighbor == -1:
          break
        i_neighbor,j_neighbor = get_ij(neighbor,num_row,num_col)
        i_node,j_node = get_ij(node,num_row,num_col)
        #Checking which boundary this is on.
        if i_neighbor == i_node:
          bounds[1] = True
        else:
          bounds[0] = True
      
      #Determining the number of boundary cells to add to the cost.
      boundary_cells = 0.0
      comm_overhead = 0.0
      #We first see how many True entries are in bounds.
      num_true = sum(bounds)
      if num_true == 2:
        boundary_cells = max(bdy_cells_per_subset[node])
        comm_overhead = latency*m_l
      elif num_true == 1:
        #Communicating across an x-boundary only.
        if  bounds[0] == True:
          boundary_cells = bdy_cells_per_subset[node][1]
        #Communicating across a y-boundary only.
        elif bounds[1] == True:
          boundary_cells = bdy_cells_per_subset[node][0]
      
      #Cells in this subset.
      num_cells = cells_per_subset[node]
      
      #The cost of this edge. upc = uknowns per cell
      #upbc = unkowns per boundary cell
      cost = num_cells*t_u*upc + (boundary_cells*upbc*t_comm + comm_overhead)
      graph[e[0]][e[1]]['weight'] = cost
    
  for ig in range(0,num_graphs):
    graph = graphs[ig]
    for n in range(0,num_subsets):
      out_edges = list(graph.out_edges(n,'weight'))
      num_edges = len(out_edges)
      out_edges = [out_edges[i][2] for i in range(num_edges)]
      time_to_solve[ig][n] = max(out_edges)
  
  return graphs,time_to_solve

#This inverts the weights of a graph in order to be able to calculate the true longest path.
def invert_weights(graph):
  
  for u,v,d in graph.edges(data=True):
    d['weight'] = -1.0*d['weight']
  
  return graph

#Converts the edge weighting to a universal time. An edge will now represent the time on a universal scale. For instance, if we have an edge that starts at node A and ends at node B, the edge represents the total time it takes that path to get to node B. 
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

#A weight based traversal of a graph G. In the context of our problem, this returns all nodes solving at time t = weight_limit.
def nodes_being_solved_simple(G,prev_nodes,weight_limit,time_to_solve):
  #The ending node of the graph is always -1. 
  end_node =  -1
  
  #The nodes being solved.
  nodes_being_solved = []
  #Looping over the previous nodes and getting the simple paths to the ending node.
  for prev_node in prev_nodes:
    
    node_generator = all_simple_paths_modified(G,prev_node,end_node,time_to_solve,cutoff=weight_limit)
    nodes_being_solved += list(node_generator)

  #Making the list unique.
  nodes_being_solved = list(set(nodes_being_solved))
  nodes_being_solved = sorted(nodes_being_solved)
  return nodes_being_solved
  
def sum_weights_of_path(graph,path):
  weight_sum = 0.0
  path_length = len(path) - 1
  for n in range(0,path_length):
    node1 = path[n]
    node2 = path[n+1]
    weight = graph[node1][node2]['weight']
    weight_sum += weight

  return weight_sum

#An attempt to speep up nodes_being_solved (def on line 218).

def nodes_being_solved_faster(G,prev_nodes,weight_limit,time_to_solve):
  #The ending node of the graph is always -1. 
  end_node =  -1
  
  #The nodes being solved.
  nodes_being_solved = []
  #Looping over the previous nodes and getting the simple paths to the ending node.
  for prev_node in prev_nodes:
    
    simple_paths = nx.all_simple_paths(G,prev_node,end_node)
    #Looping over the simple paths from the previous node to the ending node.
    for path in simple_paths:
      #Number of nodes in this path
      num_nodes_path = len(path)
      for n in range(1,num_nodes_path):
        node1 = path[n-1]
        node2 = path[n]
        current_weight = G[node1][node2]['weight']
        #Checking if the node is solving by the weight limit.
        if ( current_weight >= weight_limit):
          #The time this node starts solving at.
          try:
            start_time = list(G.in_edges(node1,'weight'))[0][2]
          except:
            start_time = 0.0
          #If the start time of the node has already passed t, we break out of the loop.
          if (start_time > weight_limit):
            break
          #Is this node is actually being solved? Or just waiting to communicate?
          elif (start_time+time_to_solve[node1] > weight_limit):
            #The node is actually being solved.
            nodes_being_solved.append(node1)
            break
          
  #Making the list unique.
  nodes_being_solved = list(set(nodes_being_solved))
  nodes_being_solved = sorted(nodes_being_solved)
  return nodes_being_solved

#Returns the heaviest path to a node.
def get_heaviest_path_faster(graph,start_node,target_node):
  
  #A graph with the weights inverted. We use this to calculate the longest path.
  #inverse_graph = deepcopy(graph)
  graph = invert_weights(graph)
  
  heaviest_path = nx.bellman_ford_path(graph,start_node,target_node,weight='weight')
  #Returning the weights to normal.
  graph = invert_weights(graph)
  #Getting the path length of the heaviest path.  
  heaviest_path_weight = sum_weights_of_path(graph,heaviest_path)
  
  return heaviest_path_weight
  
      
#Returns the depth of graph remaining.
def get_DOG_remaining(graph):

  #Getting the final sweep time for this graph.
  final_sweep_time = list(graph.in_edges(-1,'weight'))[0][2]

  return final_sweep_time

#Sorts path indices based on priority octants.
def sort_priority(graph_indices):
  #The true octant priorities.
  true_priorities = [0,4,1,5,2,6,3,7]
  original_graph_indices = copy(graph_indices)
  test_indices = []
  for p in range(0,len(graph_indices)):
    graph_index = graph_indices[p]
    test_indices.append(true_priorities.index(graph_index))
    
    
  test_indices = sorted(test_indices)
  index_map = {}
  for i in range(0,len(test_indices)):
    graph_index = test_indices[i]
    graph_indices[i] = true_priorities[graph_index]
    old_index = original_graph_indices.index(graph_indices[i])
    index_map[i] = old_index
    
  return graph_indices


#Takes simple paths for a graph and dumps them out.
def print_simple_paths(path_gen):
  for p in path_gen:
    print(p)

#Finds the next time where a node is ready to solve. This is the time where we solve for conflicts.
def find_next_interaction(graphs,prev_nodes,start_time,time_to_solve):
  
  num_graphs = len(graphs)
  
  next_time = float("inf")
  for g in range(0,num_graphs):
    
    graph = graphs[g]
    #Getting the nodes being solved at the start time.
    start_nodes = prev_nodes[g]
    end_node = -1
    
    for node in start_nodes:
      #Getting all the paths froward from this node till the end.
      simple_paths = nx.all_simple_paths(graph,node,end_node)
      #Getting the next time of interaction (when the next node is ready to solve)
      for path in simple_paths:
        next_node = path[1]
        #Getting the time the next node is ready to solve.
        next_node_solve = graph[node][next_node]['weight']
        if next_node_solve < next_time:
          next_time = next_node_solve
     
  return next_time

def find_next_interaction_simple(graphs,prev_nodes,start_time,time_to_solve):  
  num_graphs = len(graphs)
  
  next_time = float("inf")
  for g in range(0,num_graphs):
    
    graph = graphs[g]
    #Getting the nodes being solved at the start time.
    start_nodes = prev_nodes[g]
    
    for node in start_nodes:
      successors = list(graph.successors(node))
      for s in successors:
        #Getting the time the next node is ready to solve.
        next_node_solve = graph[node][s]['weight']
        if next_node_solve < next_time:
          next_time = next_node_solve
  
  return next_time

#Checks if there are conflicts.
def find_conflicts(nodes):
  
  num_graphs = len(nodes)
  
  #A dict storing the conflicting graphs for each node that is in conflict.
  conflicting_nodes = {}
  for g in range(0,num_graphs-1):
    #The current graph's conflicting nodes.
    primary_nodes = nodes[g]
    for g2 in range(g+1,num_graphs):
      secondary_nodes = nodes[g2]
      
      nodes_in_conflict = list(set(primary_nodes) & set(secondary_nodes))
      num_nodes_in_conflict = len(nodes_in_conflict)
      
      for n in range(0,num_nodes_in_conflict):
        
        node = nodes_in_conflict[n]
        try:
          conflicting_nodes[node].append(g)
        except:
          conflicting_nodes[node] = [g]
          
        conflicting_nodes[node].append(g2)
        #Making the list unique.
        conflicting_nodes[node] = list(set(conflicting_nodes[node]))
  
  return conflicting_nodes

#Finds the first conflict in a group of conflicting nodes. This will affect downstream nodes.
def find_first_conflict(conflicting_nodes,graphs):
  
  #A list to store the nodes that are ready to solve at the exact same time.
  first_nodes = []
  min_ready_to_solve = float("inf")
  for n in conflicting_nodes:
    #Which graphs are conflicting on this node.
    conflicting_graphs = conflicting_nodes[n]
    num_conflicting_graphs = len(conflicting_graphs)
    #Looping through the conflicting graphs.
    for g in range(0,num_conflicting_graphs):
      graph = graphs[conflicting_graphs[g]]
      #Getting the inbound edges to our node.
      in_edges_n = list(graph.in_edges(n,'weight'))
      #If there are no inbound edges to the node, it means that it is the initial node in the graph. This means that it must be the first conflict.
      if not in_edges_n:
        first_node = n
        first_nodes.append(first_node)
        return first_nodes
      else:
        ready_to_solve = in_edges_n[0][2]
        if ready_to_solve < min_ready_to_solve:
          min_ready_to_solve = ready_to_solve
  
  #Now that we have the minimum solve time, we can get all the nodes ready at this time.
  
  for n in conflicting_nodes:
    #Which graphs are conflicting on this node.
    conflicting_graphs = conflicting_nodes[n]
    num_conflicting_graphs = len(conflicting_graphs)
    #Looping through the conflicting graphs.
    for g in range(0,num_conflicting_graphs):
      graph = graphs[conflicting_graphs[g]]
      #Getting the inbound edges to our node.
      in_edges_n = list(graph.in_edges(n,'weight'))
      
      #Checking what time this node is ready to solve at.
      ready_to_solve = in_edges_n[0][2]
      #If this is ready to solve at the minimum ready to solve time, it belongs in first_nodes.
      if ready_to_solve == min_ready_to_solve:
        first_nodes.append(n)
        break
  
  return first_nodes

#Finds the first graph to get to a conflicted node. In case they arrive at the same time, we return the graph that has a greater depth of graph remaining. In case of a tie with the DOG remaining, we return the graph that has the priority octant.
def find_first_graph(conflicting_graphs,graphs,node):
  
  num_conflicting_graphs = len(conflicting_graphs)
  #Stores the amount of time it takes each conflicting graph to arrive at the node.
  graph_times = [None]*num_conflicting_graphs
  #Stores the index into the list of graphs of the conflicting graphs.
  graph_indices = [None]*num_conflicting_graphs
  for g in range(0,num_conflicting_graphs):
    #The original index of the conflicting graphs.
    graph_index = conflicting_graphs[g]
    graph = graphs[graph_index]
    #The incoming edges to our node.
    in_edges = list(graph.in_edges(node,'weight'))
    #The time it takes for this graph to get to the node. If there are no edges inbound to the node, then the time to get to the node is 0.0.
    try:
      time_to_node = in_edges[0][2]
    except:
      time_to_node = 0.0
    graph_times[g] = time_to_node
    graph_indices[g] = graph_index
  
  #We pull the graph with the minimum time to node.
  min_time_to_node = min(graph_times)
  #We check if there are multiple graphs ready at the same time (no guarantee same time is the minimum time). This line will return true if there are duplicates.
  if len(graph_times) != len(set(graph_times)):
    #We check if there are multiple instances of the min_time_to_node
    min_times = []
    for g in range(0,len(graph_times)):
      if graph_times[g] == min_time_to_node:
        min_times.append(graph_indices[g])
    #If multiple graphs are ready to solve the node at the minimum time, we resort to a depth of graph remaining test.
    len_min_times = len(min_times)
    if len_min_times > 1:
      #Storing the depth of graphs remaining for all tied graphs.
      dogs_remaining = []
      #Storing the graph indices.
      dogs_graph_indices = []
      for i in range(0,len_min_times):
        #Getting the graph index of the tied graph.
        graph_index = min_times[i]
        graph = graphs[graph_index]
        #The depth of graph remaining in this graph. This is equivalent to the time it takes to sweep the graph.
        dog_remaining = get_DOG_remaining(graph)
        dogs_remaining.append(dog_remaining)
        dogs_graph_indices.append(graph_index)
      
      #We check if there are multiple graphs with the same DOG remaining.
      if len(dogs_remaining) != len(set(dogs_remaining)):
        max_dog_remaining = max(dogs_remaining)
        #Checking if the maximum depth of graph is tied or just secondary depths of graph.
        max_dogs = []
        for d in range(0,len(dogs_remaining)):
          if dogs_remaining[d] == max_dog_remaining:
            max_dogs.append(dogs_graph_indices[d])
        
        
        #Once we have the graph indices of the tied graphs (according to DOG remaining), the we check to see if they have a different lengths of shortest paths.
        max_dogs_same_length = []
        for d in range(0,len(max_dogs)):
          graph_index = max_dogs[d]
          shortest_length = len(list(nx.shortest_path(graphs[graph_index],node,-1)))
          max_dogs_same_length.append(shortest_length)
        
        #If all shortest path lengths are the same, we sort based on priority.
        if len(max_dogs_same_length) != len(set(max_dogs_same_length)):
          max_dogs = sort_priority(max_dogs)
          first_graph = max_dogs[0]
        #Otherwise we choose the one with the "longest" shortest path.
        else:
          short_path_index = max_dogs_same_length.index(max(max_dogs_same_length))
          first_graph = max_dogs[short_path_index]
          
      
      #We only need to look at the graph with the max DOG remaining. It is the first graph that should start solving.
      else:
        max_dog_remaining_index = dogs_remaining.index(max(dogs_remaining))
        #This is the first graph that will start solving.
        first_graph = dogs_graph_indices[max_dog_remaining_index]
    
    #If only one graph is ready to solve at the minimum time, that graph is our first graph.    
    else:
      min_time_to_node_index = graph_times.index(min(graph_times))
      first_graph = graph_indices[min_time_to_node_index]
  
  #We take the graph that gets to the node first and that is our first graph.
  else:
    min_time_to_node_index = graph_times.index(min(graph_times))
    first_graph = graph_indices[min_time_to_node_index]
      
  return first_graph

#This function does the same thing as modify_secondary_graphs but in the case that multiple nodes are ready to solve at time t.
def modify_secondary_graphs_mult_node(graphs,conflicting_nodes,nodes,time_to_solve):
  
  #Copying the graphs frozen at time t.
  frozen_graphs = deepcopy(graphs)
  num_graphs = len(graphs)
  num_nodes = len(nodes)
  #Storing modified edges per graph over all nodes per time t.
  modified_edges_over_nodes = [{k: [] for k in range(num_graphs)} for i in range(num_nodes+1)]
  node_ind = 1
  #We loop over all nodes ready to solve at time t.
  for node in nodes:
    #The time to solve this node.
    time_to_solve_node = time_to_solve[node]
    #We get the graphs in conflict at this node.
    conflicting_graphs = conflicting_nodes[node]
    num_conflicting_graphs = len(conflicting_graphs)
    #modified_edges = deepcopy(modified_edges_over_nodes[node_ind-1])
    for outer in range(0,num_conflicting_graphs-1):
      print(node,conflicting_graphs)
      #Storing modified edges per graph at time t.
      modified_edges = deepcopy(modified_edges_over_nodes[node_ind-1])
      #The fastest graph to the node.
      first_graph = find_first_graph(conflicting_graphs,frozen_graphs,node)
      #Removed from conflicting graphs.
      conflicting_graphs.remove(first_graph) 
      #Loop over the secondary graphs.
      for g in range(0,len(conflicting_graphs)):
        second_graph = conflicting_graphs[g]
        #The delay the second_graph will incur.
        delay = calculate_delay(first_graph,second_graph,frozen_graphs,node,time_to_solve_node)
        secondary_graph = graphs[second_graph]
        #We need to first add the delay to the preceding edges, in order to update the time this node is ready to solve at.
        edges = list(secondary_graph.in_edges(node))
        num_edges = len(edges)
        for e in range(0,num_edges):
          node1,node2 = edges[e]
          secondary_graph[node1][node2]['weight'] += delay
        
        #if (second_graph == 6):
          #print("debug stop")
        #secondary_graph = modify_downstream_edges(secondary_graph,node,-1,modified_edges[second_graph],delay)
        secondary_graph = modify_downstream_edges_faster(secondary_graph,second_graph,node,modified_edges[second_graph],time_to_solve,delay)
        #if (second_graph == 6):
          #print("debug stop")
      
        graphs[second_graph] = secondary_graph
    
    modified_edges_over_nodes[node_ind] = modified_edges
    node_ind += 1
      
  #Make sure all incoming edges to all nodes match up. We do this later in the multi-node case because this all occurs within one time iteration.
  for g in range(0,num_graphs):
    graphs[g] = match_delay_weights(graphs[g])
  
  return graphs

#This function does the same thing as modify_secondary_graphs but in the case that multiple nodes are ready to solve at time t.
def modify_secondary_graphs_mult_node_improved(graphs,conflicting_nodes,nodes,time_to_solve):
  
  #Copying the graphs frozen at time t.
  frozen_graphs = deepcopy(graphs)
  num_graphs = len(graphs)
  num_nodes = len(nodes)
  #Storing modified edges per graph over all nodes per time t.
  modified_edges_over_nodes = [{k: [] for k in range(num_graphs)} for i in range(num_nodes+1)]
  node_ind = 1
  #We loop over all nodes ready to solve at time t.
  for node in nodes:
    #We get the graphs in conflict at this node.
    conflicting_graphs = conflicting_nodes[node]
    print(node,conflicting_graphs)
    #Storing modified edges per graph at time t.
    modified_edges = deepcopy(modified_edges_over_nodes[node_ind-1])
    if node == 1:
      print("debug stop")
    #The fastest graph to the node.
    first_graph = find_first_graph(conflicting_graphs,frozen_graphs,node)
    #Removed from conflicting graphs.
    conflicting_graphs.remove(first_graph) 
    #Loop over the secondary graphs.
    for g in range(0,len(conflicting_graphs)):
      second_graph = conflicting_graphs[g]
      #The delay the second_graph will incur.
      delay = calculate_delay(first_graph,second_graph,frozen_graphs,node,time_to_solve[first_graph][node])
      secondary_graph = graphs[second_graph]
      if delay > 0.0:
        #We need to first add the delay to the preceding edges, in order to update the time this node is ready to solve at.
        edges = list(secondary_graph.in_edges(node))
        num_edges = len(edges)
        for e in range(0,num_edges):
          node1,node2 = edges[e]
          secondary_graph[node1][node2]['weight'] += delay
        
        #secondary_graph = modify_downstream_edges(secondary_graph,node,-1,modified_edges[second_graph],delay)
        secondary_graph = modify_downstream_edges_faster(secondary_graph,second_graph,node,modified_edges[second_graph],time_to_solve,delay)    
        graphs[second_graph] = secondary_graph
      
    modified_edges_over_nodes[node_ind] = modified_edges
    
    node_ind += 1
      
  #Make sure all incoming edges to all nodes match up. We do this later in the multi-node case because this all occurs within one time iteration.
  for g in range(0,num_graphs):
    graphs[g] = match_delay_weights(graphs[g])
  
  return graphs

#Modifies the weights of the secondary conflicting graphs at a particular node.
def modify_secondary_graphs(graphs,conflicting_graphs,node,time_to_solve):
  
  #The number of secondary conflicting graphs.
  num_conflicting_graphs = copy(len(conflicting_graphs))
  
  for outer in range(0,num_conflicting_graphs-1):
    #The fastest graph to the node.
    first_graph = find_first_graph(conflicting_graphs,graphs,node)
#    print("Conflicted Node: ", node)
#    print("First graph: ", first_graph)
    #Removed from conflicting graphs.
    conflicting_graphs.remove(first_graph)
    #Loop over the secondary graphs.
    for g in range(0,len(conflicting_graphs)):
      second_graph = conflicting_graphs[g]
      #The delay the second_graph will incur.
      delay = calculate_delay(first_graph,second_graph,graphs,node,time_to_solve[first_graph][node])
      #The secondary graph who's weights we are modifying.
      secondary_graph = graphs[second_graph]
      #We need to first add the delay to the preceding edges, in order to update the time this node is ready to solve at.
      edges = list(secondary_graph.in_edges(node))
      num_edges = len(edges)
      for e in range(0,num_edges):
        node1,node2 = edges[e]
        secondary_graph[node1][node2]['weight'] += delay
      #All paths from the node in conflict until the end of the graph.
      secondary_paths = nx.all_simple_paths(secondary_graph,node,-1)
      #A list of edges that have already been modified.
      modified_edges = []
      #Looping over all of downstream secondary paths.
      for path in secondary_paths:
        len_path = len(path)-1
        for n in range(0,len_path):
          node1 = path[n]
          node2 = path[n+1]
          #The edge.
          edge = (node1,node2)
          #Checking if this edge has already been modified. If it has, we DO NOT need to modify it again.
          if (edge not in modified_edges):
            #Adding the delay. 
            secondary_graph[node1][node2]['weight'] += delay
            #Adding this edge to the modified edges.
            modified_edges.append(edge)
      
      #Make sure all incoming edges to all nodes match up.
      secondary_graph = match_delay_weights(secondary_graph)
      
      graphs[second_graph] = secondary_graph

  return graphs

def match_delay_weights(graph):
  
  num_nodes = graph.number_of_nodes()-1
  
  for n in range(0,num_nodes):
    
    #The incoming edges to this node.
    edges = list(graph.in_edges(n,'weight'))
    num_edges = len(edges)
    
    #If the number of incoming edges is greater than one, we need to match the weights.
    if num_edges > 1:
      
      #Get the weights of the edges.
      weights = [z for x,y,z in edges]
      #The maximum weight.
      max_weight = copy(max(weights))
      
      #Looping through the edges 
      for e in range(0,num_edges):
        node1,node2,weight = edges[e]
        if weight == max_weight:
          continue
        else:
          graph[node1][node2]['weight'] = max_weight
      
  return graph   
  
#Calculates the delay that is incurred to a secondary graph by the winning graph at the node.
def calculate_delay(first_graph,second_graph,graphs,node,time_to_solve_node):
  #The start time of the first graph.
  first_start_time = 0.0
  try:
    first_start_time = list(graphs[first_graph].in_edges(node,'weight'))[0][2]
  except:
    first_start_time = 0.0
  
  #The start time of the second graph.
  second_start_time = 0.0
  try:
    second_start_time = list(graphs[second_graph].in_edges(node,'weight'))[0][2]
  except:
    second_start_time = 0.0
  
  #The delay is the difference in start times.
  delay = first_start_time + time_to_solve_node - second_start_time
  #If this delay value is less than zero, then the first graph has finished solving it by the time the second graph gets to it.
  if delay < 0.0 or delay < 10e-12:
    delay = 0.0
  
  return delay
    
def add_conflict_weights(graphs,time_to_solve):
  
  #The number of graphs.
  num_graphs = len(graphs)
  
  #Storing the ending nodes of all graphs.
  end_nodes = {}
  starting_nodes = []
  prev_nodes = []
  for g in range(0,num_graphs):
    graph = graphs[g]
    end_nodes[g] = list(graph.predecessors(-1))[0]
    prev_nodes.append([x for x in graph.nodes() if graph.in_degree(x) == 0])
    starting_nodes.append(deepcopy(prev_nodes[g]))
  
  #The number of graphs that have finished.
  num_finished_graphs = 0
  #The list of finished graphs.
  finished_graphs = [False]*num_graphs
  #The current time (starts at 0.0 s)
  t = 0.0
  
  #Keep iterating until all graphs have finished.
  while num_finished_graphs < num_graphs:
    print('Time t = ', t)
    #Getting the nodes that are being solved at time t for all graphs.
    all_nodes_being_solved = [None]*num_graphs
    for g in range(0,num_graphs):
      graph = graphs[g]
      if not prev_nodes[g]:
        prev_nodes[g] = starting_nodes[g]
      #all_nodes_being_solved[g] = nodes_being_solved_faster(graph,prev_nodes[g],t,time_to_solve)
      all_nodes_being_solved[g] = nodes_being_solved_simple(graph,prev_nodes[g],t,time_to_solve[g])
      
    prev_nodes = all_nodes_being_solved
    print("Nodes being solved in each graph")
    print(all_nodes_being_solved)
    #Finding any nodes in conflict at time t.
    conflicting_nodes = find_conflicts(all_nodes_being_solved)
    num_conflicting_nodes = len(conflicting_nodes)
    
    print("The graphs in conflict for each node")
    print(conflicting_nodes)
    #If no nodes are in conflict, we continue to the next interaction.
    if bool(conflicting_nodes) == False:
      #t = find_next_interaction(graphs,prev_nodes,t,time_to_solve)
      t = find_next_interaction_simple(graphs,prev_nodes,t,time_to_solve)

    #Otherwise, we address the conflicts between nodes across all graphs.
    else:
      #Find nodes ready to solve at time t that are in conflict.
      first_nodes = find_first_conflict(conflicting_nodes,graphs)
      print(first_nodes)
#      num_nodes_ready_to_solve = len(first_nodes)
#      if (num_nodes_ready_to_solve == 1):
#        first_node = first_nodes[0]
#        #The conflicting grpahs at this node.
#        conflicting_graphs = conflicting_nodes[first_node]
#        #We need to modify the weights of the secondary graphs. This function will find the "winning" graph and modify everything downstream in losing graphs.
#        graphs = modify_secondary_graphs(graphs,conflicting_graphs,first_node,time_to_solve)
#        #To update our march through, we need to update t here, with a find_next_interaction.
#        if (num_conflicting_nodes == 1):
#          #t = find_next_interaction(graphs,prev_nodes,t,time_to_solve)
#          t = find_next_interaction_simple(graphs,prev_nodes,t,time_to_solve)
#      
#      else:
        #We need to modify the weights of the secondary graphs. This function will find the "winning" graph and modify everything downstream in losing graphs.
      graphs = modify_secondary_graphs_mult_node_improved(graphs,conflicting_nodes,first_nodes,time_to_solve)
      #To update our march through, we need to update t here, with a find_next_interaction.
      if (num_conflicting_nodes == len(first_nodes)):
        t = find_next_interaction_simple(graphs,prev_nodes,t,time_to_solve)

#    plot_graphs(graphs,t)
#    print("here")
    #Checking if any of the graphs have finished.
    for g in range(0,num_graphs):
      if finished_graphs[g]:
        continue
      #The end node of this graph.
      end_node = end_nodes[g]
      #The time it takes to finish this graph.
      time_to_finish = graphs[g][end_node][-1]['weight']
      #If the current universal time is greater than the time to finish sweeping the graph, we say this graph is finished.
      if t >= time_to_finish:
        finished_graphs[g] = True
    
    #print(finished_graphs)
    num_finished_graphs = len([x for x in finished_graphs if finished_graphs[x] == True])

    
  return graphs


def compute_solve_time(graphs):
  
  num_graphs = len(graphs)
  solve_times = [None]*num_graphs
  for g in range(0,num_graphs):
    graph = graphs[g]
    solve_times[g] = list(graph.in_edges(-1,'weight'))[0][2]

  max_time = max(solve_times)
  return solve_times,max_time

#Get the y_cuts from global subset boundaries.x
def get_y_cuts(boundaries,numrow,numcol):
  
  y_cuts = [[None]*(numrow+1) for col in range(numcol)]
  #Looping through columns.
  for col in range(0,numcol):
    for row in range(0,numrow+1):
      if (row == numrow):
        ss_id = get_ss_id(col,row-1,numrow)
        y_cuts[col][row] = boundaries[ss_id][3]
      else:
        ss_id = get_ss_id(col,row,numrow)
        y_cuts[col][row] = boundaries[ss_id][2]
      
  return y_cuts
      
#The driving function to compute the time to solution.
def time_to_solution(f,subset_bounds,machine_params,num_col,num_row):
  
  #Getting mesh information.
  cells_per_subset, bdy_cells_per_subset = get_cells_per_subset_2d(f,subset_bounds)
  #Getting the y_cuts in order to build the adjacency matrix.
  y_cuts = get_y_cuts(subset_bounds,num_row,num_col)
  #Building the adjacency matrix.
  adjacency_matrix = build_adjacency(subset_bounds,num_col-1,num_row-1,y_cuts)
  #Building the graphs.
  graphs = build_graphs(adjacency_matrix,num_row,num_col)
  #Weighting the graphs with the preliminary info of the cells per subset and boundary cells per subset. This will also return the time to solve each subset.
  graphs,time_to_solve = add_edge_cost(graphs,subset_bounds,cells_per_subset,bdy_cells_per_subset,machine_params,num_row,num_col)
  #Making the edges universal.
  graphs = make_edges_universal(graphs)
  plot_graphs(graphs,0)
  #Adding delay weighting.
  graphs = add_conflict_weights(graphs,time_to_solve)
  #plot_graphs(graphs,0)
  solve_times,max_time = compute_solve_time(graphs)
  return max_time
  