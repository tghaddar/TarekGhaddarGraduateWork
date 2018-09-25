import numpy as np
import warnings
import networkx as nx
import copy
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
  
  global_xy_ratio = (global_x_max - global_x_min)/(global_y_max - global_y_min)
  nsy = int(np.sqrt(num_mini_sub)/global_xy_ratio)
  nsx = int(num_mini_sub/nsy)
  cells_per_subset = []
  for s in range(0,num_subsets):
    #The boundaries for this node.
    bounds = global_subset_boundaries[s]
    #Ratio of x-length to y-length of the subset.
    x_ratio = (bounds[1]-bounds[0])/(global_x_max - global_x_min)
    y_ratio = (bounds[3]-bounds[2])/(global_y_max - global_y_min)
    z_ratio = (bounds[5]-bounds[4])/(global_z_max - global_z_min)
    #Approx number of mini subsets in each direction.
    num_sub_y = int(nsy*y_ratio)
    num_sub_x = int(nsx*x_ratio)
    print(num_sub_x,num_sub_y)
    cells_in_subset = 2.0*num_sub_y*num_sub_x
    cells_per_subset.append(cells_in_subset)
    
  return cells_per_subset
    
  

#Flattening the directed graph using a variant of Kahn's algorithm.
def flatten_graph(graph,successors):
  flattened_graph = []
  #Finding nodes with no incoming edges.
  starting_nodes = [x for x in graph.nodes() if graph.in_degree(x)==0]
  st_nodes_copy = copy.copy(starting_nodes)
  flattened_graph.append(st_nodes_copy)
  #flattened_graph.pop(0)
  while starting_nodes:
    temp = []
    st_nodes_copy = copy.copy(starting_nodes)
    for i in range(0,len(st_nodes_copy)):
      #Getting the node to start.
      node = st_nodes_copy[i]
      #temp.append(node)
      #Successors to this node.
      node_succ = successors[node]
      #Removing this node from starting_nodes.
      starting_nodes.remove(node)
  
      #Checking which successors go next.
      for s in range(0,len(node_succ)):
        test_node = node_succ[s]
        #Removing the edge.
        graph.remove_edge(node,test_node)
        #Checking that this is the only edge coming into this node (ready to solve).
        if graph.in_degree(test_node) == 0:
          starting_nodes.append(test_node)
          temp.append(test_node)
        
    flattened_graph.append(temp)
    
  flattened_graph = [x for x in flattened_graph if x]
  return flattened_graph
    
#Checking if the current node shares x or y subset boundaries.
def find_bounds(node,succ,num_row,num_col,num_plane):
  bounds_check = []
  for s in range(0,len(succ)):
    test = succ[s]
    i_node,j_node,k_node = get_ijk(node,num_row,num_col,num_plane)
    i_succ,j_succ,k_succ = get_ijk(test,num_row,num_col,num_plane)
    
    if (i_succ != i_node):
      bounds_check.append('x')
    if (j_node == j_succ + 1 or j_node == j_succ - 1) and (i_succ == i_node):
      bounds_check.append('y')
    if(k_node != k_succ):
      bounds_check.append('z')
    
    
  return bounds_check
  

def compute_solve_time(tdgs,t_byte,m_l,cells_per_subset,global_subset_boundaries,num_row,num_col):
  time = 0
  all_graph_time = np.zeros(4)
  #Number of nodes in the graph.
  num_nodes = nx.number_of_nodes(tdgs[0])
  #The time it takes to solve a cell.
  solve_cell = 0.1
  #Looping over the graphs.
  for ig in range(0,len(tdgs)):
    #Time it takes to traverse this graph.
    time_graph = 0.0
    #The current graph
    graph = tdgs[ig]    
    #Storing the successors and predecessors for each node in the graph.
    successors = dict.fromkeys(range(num_nodes))
    predecessors = dict.fromkeys(range(num_nodes))
    for n in range(0,num_nodes):
      successors[n] = list(graph.successors(n))
      predecessors[n] = list(graph.predecessors(n))
    
    flat_graph = flatten_graph(graph,successors)
    print(flat_graph)
    for n in range(0,len(flat_graph)):
      current_nodes = flat_graph[n]
      max_time = 0.0
      for c in range(0,len(current_nodes)):
        node = current_nodes[c]
        #The number of cells for this node in the tdg.
        num_cells = cells_per_subset[node]
        #Add communication time for this node.
        #Getting the number of mini subsets we will need for this subset to have roughly 2 cells/mini sub
        num_mini_sub = num_cells/2
        #The boundaries for this node.
        bounds = global_subset_boundaries[node]
        #Ratio of x-length to y-length of the subset.
        xy_ratio = (bounds[1]-bounds[0])/(bounds[3]-bounds[2])
        #Approx number of subsets in each direction.
        num_sub_y = int(np.sqrt(num_mini_sub/xy_ratio))
        num_sub_x = int(num_mini_sub/num_sub_y)
        #Approximate number of cells along x boundaries.
        bound_cell_x = num_sub_x*2
        #Approximate number of cells along y boundaries.
        bound_cell_y = num_sub_y*2
        #Need to find out which boundaries we communicate to.
        node_succ = successors[node]
        #Checking which boundaries are shared.
        bounds_check = find_bounds(node,node_succ,num_row,num_col)
        if 'x' in bounds_check: 
          time_graph += bound_cell_x*solve_cell
        if 'y' in bounds_check:
          time_graph += bound_cell_y*solve_cell
          
        #Computing the time it would take to solve this node.
        temp_time = num_cells*solve_cell
        #In regular partitions, two cells may solve at once, we take the max time.
        if (temp_time > max_time):
          max_time = temp_time
      time_graph += max_time
      
    all_graph_time[ig] = time_graph  
    time = np.average(all_graph_time)
  return time
    