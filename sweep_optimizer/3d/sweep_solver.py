import numpy as np
import warnings
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import copy
from utilities import get_ijk
from math import isclose
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
T_m = 35.0
T_g = 60.0

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

def add_edge_cost(graphs,num_total_cells,global_subset_boundaries,cell_dist,t_u,upc, upbc,t_comm,latency,m_l,num_row,num_col,num_plane):
    
  num_graphs = len(graphs)

  
  for ig in range(0,num_graphs):
    graph = graphs[ig]
    for e in graph.edges():
      #The starting node of this edge.
      node = e[0]
      #Cells in this subset.
      num_cells = cell_dist[node]
      
      boundary_cells = pow(num_cells,2/3)
#      #Communicating across the z plane.
#      if (bounds_check == 'xy'):
#        boundary_cells = bound_cell_x*bound_cell_y
#      #Communicating across the y plane.
#      if (bounds_check == 'xz'):
#        boundary_cells = bound_cell_x*bound_cell_z
#      #Communicating across the x plane.
#      if(bounds_check == 'yz'):
#        boundary_cells = bound_cell_y*bound_cell_z
#      
      
      #The cost of this edge. upc = uknowns per cell
      #upbc = unkowns per boundary cell
      cost = num_cells*t_u*upc + (boundary_cells*upbc*t_comm + latency*m_l)
      graph[e[0]][e[1]]['weight'] = cost
  return graphs

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
    heavy_path_lengths = [None]*num_nodes
    #Looping over nodes to get the longest path to each node.
    for n in range(0,num_nodes):
      
      #Getting all simple paths to the node.
      simple_paths = nx.all_simple_paths(graph,start_node,n)
      #The heaviest path and the length of the heaviest path.
      heaviest_path,heaviest_path_length = get_heaviest_path(graph,simple_paths)

      #Storing this value in heavy_path_lengths.
      heavy_path_lengths[n] = heaviest_path_length
      
    #Storing the heavy path lengths as the weight value to all preceding edges.
    for n in range(0,num_nodes):
      
      #The starting node has no preceding edges so we skip it.
      if (n != start_node):
        #Getting the weight we want for preceding edges.
        new_weight = heavy_path_lengths[n]
        #Getting the predecessors to this node in the graph.
        predecessors = list(graph.predecessors(n))
        num_pred = len(predecessors)
        for p in range(0,num_pred):
          pred = predecessors[p]
          graph[pred][n]['weight'] = new_weight
    
    #Adding the value of the last edge (end_node to the dummy -1 node).
    true_end_node = list(graph.predecessors(-1))[0]
    pred_end_node = list(graph.predecessors(true_end_node))[0]
    graph[true_end_node][-1]['weight'] += graph[pred_end_node][true_end_node]['weight']
    
    graphs[g] = graph
  return graphs

#A weight based traversal of a graph G. In the context of our problem, this returns all nodes solving at time t = weight_limit.
def nodes_being_solved(G,weight_limit,time_to_solve):
  #starting_node
  start_node = [x for x in G.nodes() if G.in_degree(x) == 0][0]
  #ending_node 
  end_node =  [x for x in G.nodes() if G.out_degree(x) == 0][0]

  #A list to store the nodes that are being solved at time t = weight_limit.
  nodes_being_solved = []
  #The simple paths of this graph.
  simple_paths = nx.all_simple_paths(G,start_node,end_node)
  
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
  
def sum_weights_of_path(graph,path):
  weight_sum = 0.0
  for n in range(0,len(path)-1):
    node1 = path[n]
    node2 = path[n+1]
    weight = graph[node1][node2]['weight']
    weight_sum += weight

  return weight_sum

#Takes all simple paths in a graph and returns the heaviest one.
def get_heaviest_path(graph,paths):
  heaviest_path = 0
  heaviest_path_weight = 0.0
  for path in paths:
    path_weight = sum_weights_of_path(graph,path)
    if path_weight > heaviest_path_weight:
      heaviest_path = path
      heaviest_path_weight = path_weight
    
  return heaviest_path,heaviest_path_weight
      
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
def find_next_interaction(graphs,start_time,time_to_solve):
  
  num_graphs = len(graphs)
  
  next_time = float("inf")
  for g in range(0,num_graphs):
    
    graph = graphs[g]
    #Getting the nodes being solved at the start time.
    start_nodes = nodes_being_solved(graph,start_time,time_to_solve)
    end_node = [x for x in graph.nodes() if graph.out_degree(x) == 0][0]
    
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
  
  first_node = -1
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
        return first_node
      else:
        ready_to_solve = in_edges_n[0][2]
        if ready_to_solve < min_ready_to_solve:
          min_ready_to_solve = ready_to_solve
          first_node = n
          
  return first_node

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
            max_dogs.append(dogs_graph_indices[g])
        
        #Once we have the graph indices of the tied graphs (according to DOG remaining), the priority octant wins.
        max_dogs = sort_priority(max_dogs)
        first_graph = max_dogs[0]
      
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

#Modifies the weights of the secondary conflicting graphs at a particular node.
def modify_secondary_graphs(graphs,conflicting_graphs,node,time_to_solve_node):
  
  #The number of secondary conflicting graphs.
  num_conflicting_graphs = copy(len(conflicting_graphs))
  
  for outer in range(0,num_conflicting_graphs-1):
    #The fastest graph to the node.
    first_graph = find_first_graph(conflicting_graphs,graphs,node)
    #Removed from conflicting graphs.
    conflicting_graphs.remove(first_graph)
    #Loop over the secondary graphs.
    for g in range(0,len(conflicting_graphs)):
      second_graph = conflicting_graphs[g]
      #The delay the second_graph will incur.
      delay = calculate_delay(first_graph,second_graph,graphs,node,time_to_solve_node)
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
      
      #Looping over all of downstream secondary paths.
      for path in secondary_paths:
        
        len_path = len(path)-1
        for n in range(0,len_path):
          node1 = path[n]
          node2 = path[n+1]
          #Adding the delay 
          secondary_graph[node1][node2]['weight'] += delay
      
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
  if delay < 0.0:
    delay = 0.0
  
  return delay
    
def add_conflict_weights(graphs,time_to_solve):
  
  #The number of graphs.
  num_graphs = len(graphs)
  
  #Storing the ending nodes of all graphs.
  end_nodes = {}
  for g in range(0,num_graphs):
    graph = graphs[g]
    end_nodes[g] = list(graph.predecessors(-1))[0]
  
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
      all_nodes_being_solved[g] = nodes_being_solved(graph,t,time_to_solve)
    
    
    print(all_nodes_being_solved)
    #Finding any nodes in conflict at time t.
    conflicting_nodes = find_conflicts(all_nodes_being_solved)
    num_conflicting_nodes = len(conflicting_nodes)
    
    print(conflicting_nodes)
    
    #If no nodes are in conflict, we continue to the next interaction.
    if bool(conflicting_nodes) == False:
      t = find_next_interaction(graphs,t,time_to_solve)
    #Otherwise, we address the conflicts between nodes across all graphs.
    else:
      #Find first conflict.
      first_node = find_first_conflict(conflicting_nodes,graphs)
      #The conflicting grpahs at this node.
      conflicting_graphs = conflicting_nodes[first_node]
      #We need to modify the weights of the secondary graphs. This function will find the "winning" graph and modify everything downstream in losing graphs.
      graphs = modify_secondary_graphs(graphs,conflicting_graphs,first_node,time_to_solve[first_node])
      #To update our march through, we need to update t here, with a find_next_interaction.
      if (num_conflicting_nodes == 1):
        t = find_next_interaction(graphs,t,time_to_solve)
      
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
    
    print(finished_graphs)
    
    num_finished_graphs = len([x for x in finished_graphs if finished_graphs[x] == True])
    
  return graphs


def compute_solve_time(graphs,cells_per_subset,t_u,upc,global_subset_boundaries,num_row,num_col,num_plane):
  time = 0
  all_graph_time = np.zeros(len(graphs))
  heaviest_paths = []
  #Looping over the graphs.
  for ig in range(0,len(graphs)):
    time_graph = 0
    graph = graphs[ig]
    copy_graph = copy(graph)
    start_node = [x for x in copy_graph.nodes() if copy_graph.in_degree(x) == 0][0]
    end_node = [x for x in copy_graph.nodes() if copy_graph.out_degree(x) == 0][0]
    
    paths = nx.all_simple_paths(graph,start_node,end_node)

    heaviest_path,path_weight = get_heaviest_path(graph,paths)
    heaviest_paths.append(heaviest_path)
    time_graph = path_weight #+ t_u*upc*cells_per_subset[end_node]
    all_graph_time[ig] = time_graph 
  
  time = np.average(all_graph_time)
  return all_graph_time,time,heaviest_paths
    