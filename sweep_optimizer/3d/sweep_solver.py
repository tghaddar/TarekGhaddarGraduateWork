import numpy as np
import warnings
import networkx as nx
from copy import copy
from utilities import get_ijk
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

def add_edge_cost(graphs,num_total_cells,global_subset_boundaries,cell_dist,solve_cell,t_comm,num_row,num_col,num_plane):
    
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
  
  for ig in range(len(graphs)):
    graph = graphs[ig]
    for e in graph.edges():
      #The starting node of this edge.
      node = e[0]
      #The ending node of this edge.
      succ = e[1]
      #Finding the bounds shared by 
      bounds_check = find_shared_bound(node,succ,num_row,num_col,num_plane)
      #Bounds of the current node.
      bounds = global_subset_boundaries[node]    
      #Getting boundary cells for each possible plane.
      x_ratio = (bounds[1] - bounds[0])/a
      y_ratio = (bounds[3] - bounds[2])/b
      z_ratio = (bounds[5] - bounds[4])/c
      bound_cell_x = 2.0*nsx*x_ratio
      bound_cell_y = 2.0*nsy*y_ratio
      bound_cell_z = 2.0*nsz*z_ratio
      boundary_cells = 0.0
      #Communicating across the z plane.
      if (bounds_check == 'xy'):
        boundary_cells = bound_cell_x*bound_cell_y
      #Communicating across the y plane.
      if (bounds_check == 'xz'):
        boundary_cells = bound_cell_x*bound_cell_z
      #Communicating across the x plane.
      if(bounds_check == 'yz'):
        boundary_cells = bound_cell_y*bound_cell_z
      
      #Cells in this subset.
      num_cells = cell_dist[node]
      #The cost of this edge.
      cost = num_cells*solve_cell + boundary_cells*t_comm
      graph[e[0]][e[1]]['weight'] = cost
  return graphs
      
def sum_weights_of_path(graph,path):
  weight_sum = 0.0
  for n in range(0,len(path)-1):
    node1 = path[n]
    node2 = path[n+1]
    weight = graph[node1][node2]['weight']
    weight_sum += weight

  return weight_sum      
          
  
def compute_solve_time(graphs,solve_cell,cells_per_subset,num_cells,global_subset_boundaries,num_row,num_col,num_plane):
  time = 0
  all_graph_time = np.zeros(8)

  #Looping over the graphs.
  for ig in range(0,len(graphs)):
    time_graph = 0
    graph = graphs[ig]
    start_node = [x for x in graph.nodes() if graph.in_degree(x) == 0][0]
    end_node = [x for x in graph.nodes() if graph.out_degree(x) == 0][0]
    
    paths = nx.all_simple_paths(graph,start_node,end_node)
    heaviest_path = 0.0
    for path in paths:
      path_weight = sum_weights_of_path(graph,path)
      if path_weight > heaviest_path:
        heaviest_path = path_weight
    
    time_graph = path_weight + solve_cell*cells_per_subset[end_node]
    all_graph_time[ig] = time_graph  
  
  time = np.average(all_graph_time)
  return all_graph_time,time
    