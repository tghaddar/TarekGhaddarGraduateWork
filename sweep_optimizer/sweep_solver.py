import numpy as np
import warnings
import networkx as nx
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

def compute_solve_time(tdgs,t_byte,m_l,cells_per_subset,global_subset_boundaries,num_row,num_col):
  time = 0
  #Number of nodes in the graph.
  num_nodes = nx.number_of_nodes(tdgs[0])
  #The time it takes to solve a cell.
  solve_cell = 0.1
  #avg_triangle_area
  tri_area = 0.2
  #Looping over the graphs.
  for ig in range(0,len(tdgs)):
    #Tallying solved nodes.
    solved = []
    #Time it takes to traverse this graph.
    time_graph = 0.0
    #The current graph
    graph = tdgs[ig]
    
    #Getting the starting node for this graph.
    starting_node = [x for x in graph.nodes() if graph.in_degree(x)==0][0]
    
    #Storing the successors and predecessors for each node in the graph.
    successors = dict.fromkeys(range(num_nodes))
    predecessors = dict.fromkeys(range(num_nodes))
    for n in range(0,num_nodes):
      successors[n] = list(graph.successors(n))
      predecessors[n] = list(graph.predecessors(n))
    
    time_graph += cells_per_subset[starting_node]*solve_cell
    solved.append(starting_node)
    #The number of cells in the starting node in the tdg.
    num_cells = cells_per_subset[starting_node]
    #Getting the number of mini subsets we will need for this subset to have roughly 2 cells/mini sub
    num_mini_sub = num_cells/2
    #The boundaries for this node.
    bounds = global_subset_boundaries[starting_node]
    #Ratio of x-length to y-length of the subset.
    xy_ratio = (bounds[1]-bounds[0])/(bounds[3]-bounds[2])
    num_sub_y = int(np.sqrt(num_mini_sub/xy_ratio))
    num_sub_x = int(num_mini_sub/num_sub_y)
    #Approximate number of cells along x boundaries.
    bound_cell_x = num_sub_x*2
    #Approximate number of cells along y boundaries.
    bound_cell_y = num_sub_y*2
    #Finding the successors of this node.
    node_suc = successors[starting_node]
    
    #Adding the two communication times for each boundary.
    time_graph += bound_cell_x*solve_cell
    time_graph += bound_cell_y*solve_cell
    
    #Starting the sweep.
    num_suc = len(node_suc)
    for n in range(0,num_suc):
      node = node_suc[n]
      #The predecessors to this node.
      preds_of_node = predecessors[node]
      #Checking if all predecessors of this node have been solved.
      #Boolean value
      in_list = 0
      for p in range(0,len(preds_of_node)):
        pred = preds_of_node[p]
        if pred in solved:
          in_list += 1
      #All predecessors have been solved.
      if (in_list == len(preds_of_node)):
        time_graph += cells_per_subset[node]*solve_cell
        solved.append(node)
      else:
        
        
  return time
    