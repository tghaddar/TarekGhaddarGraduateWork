import numpy as np
import warnings
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import copy
import build_adjacency_matrix as bam
import build_3d_adjacency as b3a
from build_global_subset_boundaries import build_global_subset_boundaries
from copy import deepcopy
from utilities import get_ijk
from utilities import get_ij,get_ss_id
from math import isclose
from mesh_processor import get_cells_per_subset_2d,get_cells_per_subset_2d_numerical,get_cells_per_subset_3d,get_cells_per_subset_3d_numerical,get_cells_per_subset_3d_numerical_test
from mesh_processor import get_cells_per_subset_2d_test
import time
warnings.filterwarnings("ignore", category=DeprecationWarning)

#This function computes the solve time for a sweep for each octant. 
#Params:
#A list of TDGs.
#The number of cells per subset.
#The grind time (time to solve an unknown on a machine)
#The communication time per byte. 

#Plots graphs at a specific time. Will help with debugging.
def plot_graphs(graphs,t,counter,num_angle):
  
  #A dictionary for node positions for quadrant 0.
  Q0 = {}
  Q0[-2] = [-2,0]
  Q0[0] = [0,0]
  Q0[1] = [2,1]
  Q0[2] = [2,-1]
  Q0[3] = [4, 0]
  Q0[-1] = [6,0]
  
  #A dictionary for node positions for quadrant 1.
  Q1 = {}
  Q1[-2] = [-2,0]
  Q1[1] = [0,0]
  Q1[0] = [2,-1]
  Q1[3] = [2,1]
  Q1[2] = [4,0]
  Q1[-1] = [6,0]
  
  #A dictionary for node positions for quadrant 2.
  Q2 = {}
  Q2[-2] = [6,0]
  Q2[1] = [0,0]
  Q2[0] = [2,-1]
  Q2[3] = [2,1]
  Q2[2] = [4,0]
  Q2[-1] = [-2,0]
  
  #A dictionary for node positions for quadrant 3.
  Q3 = {}
  Q3[-2] = [6,0]
  Q3[0] = [0,0]
  Q3[1] = [2,1]
  Q3[2] = [2,-1]
  Q3[3] = [4, 0]
  Q3[-1] = [-2,0]
  
  Q = [Q0,Q1,Q2,Q3]
  for angle in range(0,num_angle-1):
    Q.append(copy(Q0))
    Q.append(copy(Q1))
    Q.append(copy(Q2))
    Q.append(copy(Q3))
  
  num_graphs = len(graphs)
  grange = range(0,4)
  for g in grange:
    plt.figure(str(g)+str(t) + str(counter))
    plt.title("Time " + str(t) + " Graph " + str(g))
    edge_labels_1 = nx.get_edge_attributes(graphs[g],'weight')
    nx.draw(graphs[g],Q[g],with_labels = True,node_color='red')
    nx.draw_networkx_edge_labels(graphs[g],Q[g],edge_labels=edge_labels_1,font_size=12)
    plt.savefig("../../figures/graph_"+str(int(t))+ "_" + str(counter)+"_"+str(g)+".pdf")
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

def modify_downstream_edges_faster(G,graph_index,source,time_to_solve,og_delay):

  source_ready_to_solve = get_max_incoming_weight(G,source)
  ready_to_solve_all = {source:source_ready_to_solve}
  
  downstream_nodes = list(nx.descendants(G,source))
  num_downstream_nodes = len(downstream_nodes)
  #We get when each downstream node is ready to solve.
  other_ready_to_solve_all = {}
  for n in range(0,num_downstream_nodes):
    current_node = downstream_nodes[n]
    #Get incoming edge with the maximum weight to this node.
    other_ready_to_solve_all[current_node] = get_max_incoming_weight(G,current_node)
  
  
  #Sorting the downstream nodes in order of when they solve.
  other_ready_to_solve_all = dict(sorted(other_ready_to_solve_all.items(),key=lambda x:x[1]))
  #Adding this to the source dict.
  ready_to_solve_all.update(other_ready_to_solve_all)
  
  
  for k,val in ready_to_solve_all.items():
    #The current node.
    node = k
    #When the current node is ready to solve.
    ready_to_solve = val
    #Get outgoing edges of this node.
    out_edges = G.out_edges(node)
    for u,v in out_edges:
      if (v in downstream_nodes):
        delay = time_to_solve[graph_index][node] + ready_to_solve - G[u][v]['weight']
        if delay > 0.0:
          G[u][v]['weight'] += delay
          ready_to_solve_all[v] = get_max_incoming_weight(G,v)
  
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

def plot_subset_boundaries_2d(global_3d_subset_boundaries,num_subsets,fname):
  plt.figure(1)
  subset_centers = []
  for i in range(0,num_subsets):
  
    subset_boundary = global_3d_subset_boundaries[i]
    xmin = subset_boundary[0]
    xmax = subset_boundary[1]
    ymin = subset_boundary[2]
    ymax = subset_boundary[3]
  
    center_x = (xmin+xmax)/2
    center_y = (ymin+ymax)/2
  
    subset_centers.append([center_x, center_y])
  
    x = [xmin, xmax, xmax, xmin,xmin]
    y = [ymin, ymin, ymax, ymax,ymin]
  
    plt.plot(x,y,'b')
    #plt.text(center_x,center_y,str(i))
  
  plt.savefig(fname)
  plt.close()
  
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

def add_edge_cost(graphs,global_subset_boundaries,cells_per_subset, bdy_cells_per_subset,machine_params,num_row,num_col,test):
  
  num_graphs = len(graphs)
  num_subsets = len(global_subset_boundaries)
  #Storing the time to solve and communicate each subset.
  time_to_solve = [[None]*num_subsets for g in range(num_graphs)]
  Twu,Tc,Tm,Tg,upc,upbc,mcff,t_comm,latency,m_l = machine_params
  
  for ig in range(0,num_graphs):
    graph = graphs[ig]
    for e in graph.edges():
      #The starting node of this edge.
      node = e[0]
      #The ending node of this edge.
      neighbor = e[1]
      bounds = [False, False]
      
      i_neighbor,j_neighbor = get_ij(neighbor,num_row,num_col)
      i_node,j_node = get_ij(node,num_row,num_col)
      #Checking which boundary this is on.
      if i_neighbor == i_node:
        bounds[1] = True
      else:
        bounds[0] = True
    
      #Determining the number of boundary cells to add to the cost.
      boundary_cells = 0.0
      #Communicating across an x-boundary.
      if  bounds[0] == True:
        boundary_cells = bdy_cells_per_subset[node][1]
      #Communicating across a y-boundary.
      elif bounds[1] == True:
        boundary_cells = bdy_cells_per_subset[node][0]
      
      #boundary_cells = sum(bdy_cells_per_subset[node])
      #Cells in this subset.
      num_cells = cells_per_subset[node]
      #If this is a testing run, our cost is 1.
      cost = 0.0
      if test:
        cost = 1.0
      else:
        cost = mcff*(Twu + 2*latency*m_l + t_comm*boundary_cells*upbc + upc*num_cells*(Tc + Tm + Tg))
      graph[node][neighbor]['weight'] = cost
    
  for ig in range(0,num_graphs):
    graph = graphs[ig]
    for n in range(0,num_subsets):
      out_edges = list(graph.out_edges(n,'weight'))
      num_edges = len(out_edges)
      out_edges = [out_edges[i][2] for i in range(num_edges)]
      time_to_solve[ig][n] = max(out_edges)
      
  return graphs,time_to_solve

def add_edge_cost_3d(graphs,global_subset_boundaries,cells_per_subset, bdy_cells_per_subset,machine_params,num_row,num_col,num_plane,Am,test):
  num_graphs = len(graphs)
  num_subsets = len(global_subset_boundaries)
  #Storing the time to solve and communicate each subset.
  time_to_solve = [[None]*num_subsets for g in range(num_graphs)]
  Twu,Tc,Tm,Tg,upc,upbc,mcff,t_comm,latency,m_l = machine_params
  #Looping over graphs.
  for ig in range(0,num_graphs):
    graph = graphs[ig]
    for e in graph.edges():
      #The starting node of this edge.
      node = e[0]
      #The second node of this edge.
      neighbor = e[1]
      bounds = [False, False,False]
      #Checking where the neighbors are located.
      i_n,j_n,k_n = get_ijk(neighbor, num_row,num_col,num_plane)
      i,j,k = get_ijk(node,num_row,num_col,num_plane)
      if k_n != k:
        bounds[0] = True
      else:
        if i_n == i:
          bounds[1] = True
        else:
          bounds[2] = True
      
      boundary_cells = 0.0
      if bounds[0] == True:
        boundary_cells = bdy_cells_per_subset[node][0]
      if bounds[1] == True:
        boundary_cells = bdy_cells_per_subset[node][1]
      if bounds[2] == True:
        boundary_cells = bdy_cells_per_subset[node][2]
      
      #Cells in this subset.
      num_cells = cells_per_subset[node]
      #If this is a testing run, our cost is 1.
      cost = 0.0
      if test:
        cost = 1.0
      else:
        cost = mcff*(Twu + 3*latency*m_l + t_comm*boundary_cells*Am*upbc + upc*num_cells*(Tc + Am*(Tm + Tg)))
      graph[e[0]][e[1]]['weight'] = cost
    
    
  for ig in range(0,num_graphs):
    graph = graphs[ig]
    for n in range(0,num_subsets):
      out_edges = list(graph.out_edges(n,'weight'))
      num_edges = len(out_edges)
      out_edges = [out_edges[i][2] for i in range(num_edges)]
      time_to_solve[ig][n] = max(out_edges)
  
  return graphs,time_to_solve
      

#Offsets edge weighting for initial nodes for angular pipelining.
def pipeline_offset(graphs,num_angles,time_to_solve):
  
  num_graphs = len(graphs)
  graphs_per_angle = int(num_graphs/num_angles)
  #Getting the starting nodes for all the graphs.
  starting_nodes = [None]*num_graphs
  for ig in range(0,num_graphs):
    starting_nodes[ig] = [x for x in graphs[ig].nodes() if graphs[ig].in_degree(x) == 0][0]
    graphs[ig].add_node(-2)
    graphs[ig].add_edge(-2,starting_nodes[ig],weight=0.0)
  
  #Only modifying graphs after angle 0.
  for ig in range(graphs_per_angle,num_graphs):
    graph = graphs[ig]
    #Which angle.
    angle = int(ig/graphs_per_angle)
    starting_node = starting_nodes[ig]
    #The pipeline cost (how much we are going to add to the initial edges).
    pipeline_cost = angle*time_to_solve[ig][starting_node]
    

    graph[-2][starting_node]['weight'] += pipeline_cost
    
    graphs[ig] = graph
    
  return graphs
    
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
    #Inverting the weights of the graph.
    graph = invert_weights(graph)
    #Getting all shortest paths.
    shortest_paths = nx.johnson(graph,weight='weight')
    #Inverting the weights back.
    graph = invert_weights(graph)
    #Looping over nodes to get the longest path to each node.
    for n in range(0,num_nodes-1):
      #If the starting node is the target node we know that path length is zero.
      if (start_node == n):
        continue
      
      heaviest_path = shortest_paths[start_node][n]
      #Gets the heaviest path length.
      heaviest_path_length = sum_weights_of_path(graph,heaviest_path)
      #heaviest_path_length = get_heaviest_path_faster(graph,start_node,n)
      #Storing this value in heavy_path_lengths.
      heavy_path_lengths[n] = heaviest_path_length
      
    #Storing the heavy path lengths as the weight value to all preceding edges.
    for n in range(0,num_nodes-1):
      
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


#Loops through nodes and finds out who is ready to solve.
def nodes_being_solved_general(G,weight_limit,time_to_solve):
  
  nodes_being_solved = []
  nodes = list(G.nodes())
  nodes.remove(-2)
  
  
  for n in nodes:
    in_edges = list(G.in_edges(n,data='weight'))
    ready_to_solve = max([z for x,y,z in in_edges])
    #ready_to_solve = get_max_incoming_weight(G,n)
    if ready_to_solve > weight_limit:
      continue
    elif ready_to_solve == weight_limit:
      if n != -1:
        nodes_being_solved.append(n)
    elif ready_to_solve + time_to_solve[n] > weight_limit:
      if n != -1:
        nodes_being_solved.append(n)
  
  return sorted(nodes_being_solved)

#Attempting to speed up nodes_being_solved_general.
def nodes_being_solved_general_sped(G,weight_limit,nodes_already_solved,time_to_solve):
  nodes_being_solved = []
  nodes = list(G.nodes())
  nodes.remove(-2)
  nodes = list(set(nodes) - set(nodes_already_solved))
  print(len(nodes))

  for n in nodes:
    in_edges = list(G.in_edges(n,data='weight'))
    ready_to_solve = max([z for x,y,z in in_edges])
    #ready_to_solve = get_max_incoming_weight(G,n)
    if ready_to_solve > weight_limit:
      continue
    elif ready_to_solve == weight_limit:
      if n != -1:
        nodes_being_solved.append(n)
    elif ready_to_solve + time_to_solve[n] > weight_limit:
      if n != -1:
        nodes_being_solved.append(n)
    elif ready_to_solve + time_to_solve[n] < weight_limit:
      if n != -1:
        nodes_already_solved.append(n)
  
  return sorted(nodes_being_solved),sorted(nodes_already_solved)

#Attempting to speed up nodes_being_solved_general.
def nodes_being_solved_general_sped_up(G,weight_limit,nodes_already_solved,time_to_solve):
  nodes_being_solved = []
  nodes = list(G.nodes())
  nodes.remove(-2)
  nodes = list(set(nodes) - set(nodes_already_solved))
  
  in_edges = list(G.in_edges(nodes,data='weight'))
  reduced_edges = [(x,y,z) for x,y,z in in_edges if z <= weight_limit]
  

  for n1,n2,weight in reduced_edges:
    if weight > weight_limit:
      continue
    elif weight == weight_limit:
      if (n2 != -1) and (n2 not in nodes_being_solved):
        nodes_being_solved.append(n2)
    elif weight + time_to_solve[n2] > weight_limit:
      if (n2 != -1) and (n2 not in nodes_being_solved):
        nodes_being_solved.append(n2)
    elif weight  + time_to_solve[n2] < weight_limit:
      if (n2 != -1) and (n2 not in nodes_already_solved):
        nodes_already_solved.append(n2)
  
  return sorted(nodes_being_solved),sorted(nodes_already_solved)
  
def sum_weights_of_path(graph,path):
  weight_sum = 0.0
  path_length = len(path) - 1
  for n in range(0,path_length):
    node1 = path[n]
    node2 = path[n+1]
    weight = graph[node1][node2]['weight']
    weight_sum += weight

  return weight_sum


#Returns the heaviest path to a node.
def get_heaviest_path_faster(graph,start_node,target_node):
  
  #A graph with the weights inverted. We use this to calculate the longest path.
  graph = invert_weights(graph)
  
  heaviest_path = nx.bellman_ford_path(graph,start_node,target_node,weight='weight')
  #Returning the weights to normal.
  graph = invert_weights(graph)
  #Getting the path length of the heaviest path.  
  heaviest_path_weight = sum_weights_of_path(graph,heaviest_path)
  
  return heaviest_path_weight

#Returns the unweighted depth of graph remaining.
def get_DOG_remaining_unweighted(graph,source):
  
  dog_remaining = nx.shortest_path_length(graph,source,-1)
  #edges = nx.bfs_edges(graph,source)
  #dog_remaining = len(list(edges))
  return dog_remaining

#Returns the depth of graph remaining.
def get_DOG_remaining(graph):

  #Getting the final sweep time for this graph.
  final_sweep_time = list(graph.in_edges(-1,'weight'))[0][2]

  return final_sweep_time

#Sorts path indices based on priority octants.
def sort_priority(graph_indices,graphs_per_angle):
  #The true octant priorities.
  true_priorities = [0,4,1,5,2,6,3,7]
  test_indices = []
  true_indices = {}
  for p in range(0,len(graph_indices)):
    graph_index = graph_indices[p]
    octant_index = graph_index%graphs_per_angle
    try:
      true_indices[octant_index] += [graph_index]
    except:
      true_indices[octant_index] = [graph_index]
    test_indices.append(true_priorities.index(octant_index))
    #test_indices.append(true_priorities.index(graph_index))
  
  #Sorting the true indices.
  for key in true_indices:
    true_indices[key] = sorted(true_indices[key])
  
  test_indices = sorted(test_indices)
  test_indices = list(set(test_indices))
#  print(test_indices)
#  print(true_indices)
  new_graph_indices = []
  for i in range(0,len(test_indices)):
    octant = true_priorities[test_indices[i]]
    new_graph_indices += true_indices[octant]
    #graph_indices[i] = true_indices[octant]
    #graph_indices[i] = true_priorities[graph_index]
    
  return new_graph_indices


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

#Finds the first graph to get to a conflicted node. In case they arrive at the same time, we return the graph that has a greater depth of graph remaining. In case of a tie with the DOG remaining, we return the graph that has the priority octant. The boolean value unweighted is True if we want to use the unweighted depth of graph remaining algorithm.
def find_first_graph(conflicting_graphs,graphs,node,num_angles,unweighted):
  #Total number of graphs.
  num_graphs = len(graphs)
  #The graphs per angle.
  graphs_per_angle = int(num_graphs/num_angles)
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
        if unweighted:
          dog_remaining = get_DOG_remaining_unweighted(graph,node)
        else:
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
          max_dogs = sort_priority(max_dogs,graphs_per_angle)
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

#Finds the first graph to get to a conflicted node. In case they arrive at the same time, we return the graph that has a greater depth of graph remaining. In case of a tie with the DOG remaining, we return the graph that has the priority octant. The boolean value unweighted is True if we want to use the unweighted depth of graph remaining algorithm.
def find_first_graph_rebuild_graphs(conflicting_graphs,all_edges,node,num_angles,unweighted):
  #Total number of graphs.
  num_graphs = len(all_edges)
  new_graphs = [None]*num_graphs
  for g in range(0,num_graphs):
    new_graphs[g] = nx.DiGraph(all_edges[g])
  #The graphs per angle.
  graphs_per_angle = int(num_graphs/num_angles)
  num_conflicting_graphs = len(conflicting_graphs)
  #Stores the amount of time it takes each conflicting graph to arrive at the node.
  graph_times = [None]*num_conflicting_graphs
  #Stores the index into the list of graphs of the conflicting graphs.
  graph_indices = [None]*num_conflicting_graphs
  for g in range(0,num_conflicting_graphs):
    #The original index of the conflicting graphs.
    graph_index = conflicting_graphs[g]
    graph = new_graphs[graph_index]
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
        graph = new_graphs[graph_index]
        #The depth of graph remaining in this graph. This is equivalent to the time it takes to sweep the graph.
        if unweighted:
          dog_remaining = get_DOG_remaining_unweighted(graph,node)
        else:
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
          shortest_length = len(list(nx.shortest_path(new_graphs[graph_index],node,-1)))
          max_dogs_same_length.append(shortest_length)
        
        #If all shortest path lengths are the same, we sort based on priority.
        if len(max_dogs_same_length) != len(set(max_dogs_same_length)):
          max_dogs = sort_priority(max_dogs,graphs_per_angle)
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
def modify_secondary_graphs_mult_node_improved(graphs,conflicting_nodes,nodes,time_to_solve,num_angles,unweighted,nodes_already_solved):
  
  #Copying the graphs frozen at time t.
  frozen_graphs = deepcopy(graphs)
  num_graphs = len(graphs)
  #Storing modified edges per graph over all nodes per time t.
  node_ind = 1
  #We loop over all nodes ready to solve at time t.
  for node in nodes:
    #We get the graphs in conflict at this node.
    conflicting_graphs = conflicting_nodes[node]
    #print(node,conflicting_graphs)
    #The fastest graph to the node.
    first_graph = find_first_graph(conflicting_graphs,frozen_graphs,node,num_angles,unweighted)
    #print("FIRST GRAPH: ", first_graph)
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
        secondary_graph = modify_downstream_edges_faster(secondary_graph,second_graph,node,time_to_solve,delay)    
        graphs[second_graph] = secondary_graph
      
    
    node_ind += 1
      
  #Make sure all incoming edges to all nodes match up. We do this later in the multi-node case because this all occurs within one time iteration.
  for g in range(0,num_graphs):
    #graphs[g] = match_delay_weights(graphs[g])
    graphs[g] = match_delay_weights_test(graphs[g],nodes_already_solved[g])
  
  return graphs


def modify_secondary_graphs_mult_node_faster(graphs,conflicting_nodes,nodes,time_to_solve,num_angles,unweighted,nodes_already_solved):
  
  #Copying the graphs frozen at time t.
  
  #frozen_graphs = deepcopy(graphs)
  num_graphs = len(graphs)
  all_edges = [None]*num_graphs
  for g in range(0,num_graphs):
    all_edges[g] = copy(graphs[g].edges(data=True))
  
  #Storing modified edges per graph over all nodes per time t.
  node_ind = 1
  #We loop over all nodes ready to solve at time t.
  for node in nodes:
    #We get the graphs in conflict at this node.
    conflicting_graphs = conflicting_nodes[node]
    #print(node,conflicting_graphs)
    #The fastest graph to the node.
    first_graph = find_first_graph_rebuild_graphs(conflicting_graphs,all_edges,node,num_angles,unweighted)
    #print("FIRST GRAPH: ", first_graph)
    #Removed from conflicting graphs.
    conflicting_graphs.remove(first_graph) 
    #Loop over the secondary graphs.
    for g in range(0,len(conflicting_graphs)):
      second_graph = conflicting_graphs[g]
      #The delay the second_graph will incur.
      delay = calculate_delay_rebuild_graphs(first_graph,second_graph,all_edges,node,time_to_solve[first_graph][node])
      secondary_graph = graphs[second_graph]
      if delay > 0.0:
        #We need to first add the delay to the preceding edges, in order to update the time this node is ready to solve at.
        edges = list(secondary_graph.in_edges(node))
        num_edges = len(edges)
        for e in range(0,num_edges):
          node1,node2 = edges[e]
          secondary_graph[node1][node2]['weight'] += delay
        
        #secondary_graph = modify_downstream_edges(secondary_graph,node,-1,modified_edges[second_graph],delay)
        secondary_graph = modify_downstream_edges_faster(secondary_graph,second_graph,node,time_to_solve,delay)    
        graphs[second_graph] = secondary_graph
      
    
    node_ind += 1
      
  #Make sure all incoming edges to all nodes match up. We do this later in the multi-node case because this all occurs within one time iteration.
  for g in range(0,num_graphs):
    #graphs[g] = match_delay_weights(graphs[g])
    graphs[g] = match_delay_weights_test(graphs[g],nodes_already_solved[g])
  
  return graphs

def match_delay_weights(graph):
  
  num_nodes = graph.number_of_nodes()-2
  
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

def match_delay_weights_test(G,nodes_already_solved):
  
  nodes = list(G.nodes())
  nodes.remove(-1)
  nodes.remove(-2)
  nodes = list(set(nodes) - set(nodes_already_solved))
  
  for n in nodes:
    in_edges = list(G.in_edges(n,data='weight'))
    max_weight = max([z for x,y,z in in_edges])
    num_edges = len(in_edges)
    
    #If the number of incoming edges is greater than one, we need to match the weights.
    if num_edges > 1:
      #Looping through the edges 
      for e in range(0,num_edges):
        node1,node2,weight = in_edges[e]
        if weight == max_weight:
          continue
        else:
          G[node1][node2]['weight'] = max_weight
  
  return G
  
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

#Calculates the delay that is incurred to a secondary graph by the winning graph at the node.
def calculate_delay_rebuild_graphs(first_graph,second_graph,all_edges,node,time_to_solve_node):
  #Total number of graphs.
  num_graphs = len(all_edges)
  new_graphs = [None]*num_graphs
  for g in range(0,num_graphs):
    new_graphs[g] = nx.DiGraph(all_edges[g])
  #The start time of the first graph.
  first_start_time = 0.0
  try:
    first_start_time = list(new_graphs[first_graph].in_edges(node,'weight'))[0][2]
  except:
    first_start_time = 0.0
  
  #The start time of the second graph.
  second_start_time = 0.0
  try:
    second_start_time = list(new_graphs[second_graph].in_edges(node,'weight'))[0][2]
  except:
    second_start_time = 0.0
  
  #The delay is the difference in start times.
  delay = first_start_time + time_to_solve_node - second_start_time
  #If this delay value is less than zero, then the first graph has finished solving it by the time the second graph gets to it.
  if delay < 0.0:
    delay = 0.0
  
  return delay
    
def add_conflict_weights(graphs,time_to_solve,num_angles,unweighted):
  
  #The number of graphs.
  num_graphs = len(graphs)
  #Storing nodes that have already been solved.
  nodes_already_solved = [[] for g in range(0,num_graphs)]
  #Storing the ending nodes of all graphs.
  end_nodes = {}
  starting_nodes = []
  prev_nodes = [[] for g in range(0,num_graphs)]
  for g in range(0,num_graphs):
    graph = graphs[g]
    end_nodes[g] = list(graph.predecessors(-1))[0]
    true_starting_node = list(graph.successors(-2))
    starting_nodes.append(copy(true_starting_node))
  
  #The number of graphs that have finished.
  num_finished_graphs = 0
  #The list of finished graphs.
  finished_graphs = [False]*num_graphs
  #The current time (starts at 0.0 s)
  t = 0.0
  
  #Keep iterating until all graphs have finished.
  counter = 0
  while num_finished_graphs < num_graphs:
#    print('Time t = ', t)
    #if (t == 0.0011634222038497584)
    #  print("debug stop")

    #Getting the nodes that are being solved at time t for all graphs.
    all_nodes_being_solved = [None]*num_graphs
    for g in range(0,num_graphs):
      graph = graphs[g]
      if not prev_nodes[g]:
        prev_nodes[g] = starting_nodes[g]
      #all_nodes_being_solved[g] = nodes_being_solved_general(graph,t,time_to_solve[g])
      all_nodes_being_solved[g],nodes_already_solved[g] = nodes_being_solved_general_sped_up(graph,t,nodes_already_solved[g],time_to_solve[g])
    prev_nodes = all_nodes_being_solved
#    print("Nodes already being solved in each graph")
#    print(nodes_already_solved)
    #Finding any nodes in conflict at time t.
    conflicting_nodes = find_conflicts(all_nodes_being_solved)
    num_conflicting_nodes = len(conflicting_nodes)
    
    #print("The graphs in conflict for each node")
    #print(conflicting_nodes)
    #If no nodes are in conflict, we continue to the next interaction.
    if bool(conflicting_nodes) == False:
      #t = find_next_interaction(graphs,prev_nodes,t,time_to_solve)
      t = find_next_interaction_simple(graphs,prev_nodes,t,time_to_solve)

    #Otherwise, we address the conflicts between nodes across all graphs.
    else:
      #Find nodes ready to solve at time t that are in conflict.
      first_nodes = find_first_conflict(conflicting_nodes,graphs)        
      #graphs = modify_secondary_graphs_mult_node_improved(graphs,conflicting_nodes,first_nodes,time_to_solve,num_angles,unweighted,nodes_already_solved)
      graphs = modify_secondary_graphs_mult_node_faster(graphs,conflicting_nodes,first_nodes,time_to_solve,num_angles,unweighted,nodes_already_solved)
      #To update our march through, we need to update t here, with a find_next_interaction.
      if (num_conflicting_nodes == len(first_nodes)):
        t = find_next_interaction_simple(graphs,prev_nodes,t,time_to_solve)

    #plot_graphs(graphs,t,counter,num_angles)
    counter += 1
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
    
    num_finished_graphs = sum(finished_graphs)

    
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

#Unpacks the parameter space from the optimizer.
def unpack_parameters(params,global_x_min,global_x_max,global_y_min,global_y_max,numcol,numrow):
  x_cuts = [global_x_min]
  y_cuts = [[global_y_min] for col in range(0,numcol)]
  
  nx = numcol-1
  ny = numrow-1
  
  
  cut_id = 0
  y_cut_id = 0
  for cut in params:
    
    if cut_id < nx:
      x_cuts.append(cut)
    elif cut_id >= nx:
      current_col = int(y_cut_id/ny)
      y_cuts[current_col].append(cut)
      y_cut_id += 1
    cut_id += 1
    
  x_cuts.append(global_x_max)
  for col in range(0,numcol):
    y_cuts[col].append(global_y_max)
    
  return x_cuts,y_cuts

def unpack_parameters_3d(params,global_x_min,global_x_max,global_y_min,global_y_max,global_z_min,global_z_max,numcol,numrow,numplane):
  
  z_cuts = [global_z_min]
  x_cuts = [[global_x_min] for plane in range(0,numplane)]
  y_cuts = [[[global_y_min] for col in range(0,numcol)] for plane in range(0,numplane)]
  
  nx = numcol-1
  ny = numrow-1
  nz = numplane-1
  
  cut_id = 0
  x_cut_id = 0

  
  for cut in params:
    
    if cut_id < nz:
      z_cuts.append(cut)
    elif cut_id >= nz:
      if cut_id < nz+numplane*nx:
        current_plane = int(x_cut_id/nx)
        x_cuts[current_plane].append(cut)
        x_cut_id += 1
      else:
        break
    
    cut_id += 1
    
  for plane in range(0,numplane):
    for col in range(0,numcol):
      for row in range(0,ny):
        y_cuts[plane][col].append(params[cut_id])
        cut_id += 1
    
  
  z_cuts.append(global_x_max)
  
  for plane in range(0,numplane):
    x_cuts[plane].append(global_x_max)
  
  for plane in range(0,numplane):
    for col in range(0,numcol):
      y_cuts[plane][col].append(global_y_max)
  
  return x_cuts,y_cuts,z_cuts
  
  

def tweak_parameters(x_cuts,y_cuts,global_x_min,global_x_max,global_y_min,global_y_max,numcol,numrow):
  
  tweak_x = (0.05/numcol)*(global_x_max - global_x_min)
  tweak_y = (0.05/numrow)*(global_y_max - global_y_min)
  
  #Quickly sorting the cut lines.
  x_cuts = sorted(x_cuts)
  for col in range(0,numcol):
    y_cuts[col] = sorted(y_cuts[col])
  
  #Tweaking the xcuts if necessary
  for cut in range(2,numcol):
    if x_cuts[cut] == x_cuts[cut-1]:
      x_cuts[cut] += tweak_x
  x_cuts = sorted(x_cuts) 
  #Tweaking the ycuts if necessary
  for col in range(0,numcol):
    y_cuts_col = y_cuts[col]
    for cut in range(2,numrow):
      if y_cuts_col[cut] == y_cuts_col[cut-1]:
        y_cuts_col[cut] += tweak_y
    
    y_cuts[col] = sorted(y_cuts_col)
  
  return x_cuts,y_cuts

def tweak_parameters_3d(x_cuts,y_cuts,z_cuts,global_x_min,global_x_max,global_y_min,global_y_max,global_z_min,global_z_max,numcol,numrow,numplane):
  
  tweak_x = (0.05/numcol)*(global_x_max - global_x_min)
  tweak_y = (0.05/numrow)*(global_y_max - global_y_min)
  tweak_z = (0.05/numplane)*(global_z_max - global_z_min)
  
  #Quickly sorting the cut lines.
  z_cuts = sorted(z_cuts)
  for plane in range(0,numplane):
    x_cuts[plane] = sorted(x_cuts[plane])
    for col in range(0,numcol):
      y_cuts[plane][col] = sorted(y_cuts[plane][col])
  
  #Tweaking the zcuts if necessary.
  for cut in range(2,numplane):
    if z_cuts[cut] == z_cuts[cut-1]:
      z_cuts[cut] += tweak_z
      
  #Tweaking the xcuts and ycuts if necessary.
  for plane in range(0,numplane):
    x_cuts_plane = x_cuts[plane]
    for cut in range(2,numcol):
      if x_cuts_plane[cut] == x_cuts_plane[cut-1]:
        x_cuts_plane[cut] += tweak_x
    x_cuts[plane] = x_cuts_plane
    
    #Tweaking the y_cuts.
    for col in range(0,numcol):
      y_cuts_col = y_cuts[plane][col]
      for cut in range(2,numrow):
        if y_cuts_col[cut] == y_cuts_col[cut-1]:
          y_cuts_col[cut] += tweak_y
      
      y_cuts[plane][col] = y_cuts_col
  
  return x_cuts,y_cuts,z_cuts

#The driving function to compute the time to solution.
def time_to_solution(f,x_cuts,y_cuts,machine_params,num_col,num_row,num_angles,test,unweighted):
  #Building subset boundaries.
  subset_bounds = build_global_subset_boundaries(num_col-1,num_row-1,x_cuts,y_cuts)
  #Getting mesh information.
  cells_per_subset, bdy_cells_per_subset = get_cells_per_subset_2d(f,subset_bounds)  
  #Building the adjacency matrix.
  adjacency_matrix = bam.build_adjacency(subset_bounds,num_col-1,num_row-1,y_cuts)
  #Building the graphs.
  graphs = bam.build_graphs(adjacency_matrix,num_row,num_col,num_angles)
  #Weighting the graphs with the preliminary info of the cells per subset and boundary cells per subset. This will also return the time to solve each subset.
  graphs,time_to_solve = add_edge_cost(graphs,subset_bounds,cells_per_subset,bdy_cells_per_subset,machine_params,num_row,num_col,test)
  graphs = pipeline_offset(graphs,num_angles,time_to_solve)
  #Making the edges universal.
  graphs = make_edges_universal(graphs)
  
  #plot_graphs(graphs,0,0,num_angles)
  #Adding delay weighting.
  graphs = add_conflict_weights(graphs,time_to_solve,num_angles,unweighted)
  #plot_graphs(graphs,0)
  solve_times,max_time = compute_solve_time(graphs)
  return max_time

#The driving function to compute the time to solution.
def time_to_solution_numerical(points,x_cuts,y_cuts,machine_params,num_col,num_row,num_angles):
  #Building subset boundaries.
  subset_bounds = build_global_subset_boundaries(num_col-1,num_row-1,x_cuts,y_cuts)
  #Getting mesh information.
  cells_per_subset, bdy_cells_per_subset = get_cells_per_subset_2d_numerical(points,subset_bounds)  
  #Building the adjacency matrix.
  adjacency_matrix = bam.build_adjacency(subset_bounds,num_col-1,num_row-1,y_cuts)
  #Building the graphs.
  graphs = bam.build_graphs(adjacency_matrix,num_row,num_col)
  #Weighting the graphs with the preliminary info of the cells per subset and boundary cells per subset. This will also return the time to solve each subset.
  graphs,time_to_solve = add_edge_cost(graphs,subset_bounds,cells_per_subset,bdy_cells_per_subset,machine_params,num_row,num_col)
  graphs = pipeline_offset(graphs,num_angles,time_to_solve)
  #Making the edges universal.
  graphs = make_edges_universal(graphs)
  

  #Adding delay weighting.
  graphs = add_conflict_weights(graphs,time_to_solve)
  #plot_graphs(graphs,0)
  solve_times,max_time = compute_solve_time(graphs)
  return max_time


#The time to solution function that is fed into the optimizer.
def optimized_tts(params,f,global_xmin,global_xmax,global_ymin,global_ymax,num_row,num_col,machine_params,num_angles,unweighted):
  
  x_cuts,y_cuts = unpack_parameters(params,global_xmin,global_xmax,global_ymin,global_ymax,num_col,num_row)
  #Building subset boundaries.
  subset_bounds = build_global_subset_boundaries(num_col-1,num_row-1,x_cuts,y_cuts)
  #Getting mesh information.
  cells_per_subset, bdy_cells_per_subset = get_cells_per_subset_2d(f,subset_bounds)  
  #Building the adjacency matrix.
  adjacency_matrix = bam.build_adjacency(subset_bounds,num_col-1,num_row-1,y_cuts)
  #Building the graphs.
  graphs = bam.build_graphs(adjacency_matrix,num_row,num_col,num_angles)
  #Weighting the graphs with the preliminary info of the cells per subset and boundary cells per subset. This will also return the time to solve each subset.
  graphs,time_to_solve = add_edge_cost(graphs,subset_bounds,cells_per_subset,bdy_cells_per_subset,machine_params,num_row,num_col,False)
  graphs = pipeline_offset(graphs,num_angles,time_to_solve)
  #Making the edges universal.
  graphs = make_edges_universal(graphs)
  
  #Adding delay weighting.
  graphs = add_conflict_weights(graphs,time_to_solve,num_angles,unweighted)
  solve_times,max_time = compute_solve_time(graphs)
  print(max_time)
  return max_time

#The time to solution function that is fed into the optimizer.
def optimized_tts_numerical(params, points,global_xmin,global_xmax,global_ymin,global_ymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted):
  start = time.time()
  machine_params = (t_u,upc,upbc,t_comm,latency,m_l)
  
  x_cuts,y_cuts = unpack_parameters(params,global_xmin,global_xmax,global_ymin,global_ymax,num_col,num_row)
  print(x_cuts,y_cuts)
  x_cuts,y_cuts = tweak_parameters(x_cuts,y_cuts,global_xmin,global_xmax,global_ymin,global_ymax,num_col,num_row)
  
  #Building subset boundaries.
  subset_bounds = build_global_subset_boundaries(num_col-1,num_row-1,x_cuts,y_cuts)
 
  #Building the adjacency matrix.
  adjacency_matrix = bam.build_adjacency(subset_bounds,num_col-1,num_row-1,y_cuts)
   #Getting mesh information.
  cells_per_subset, bdy_cells_per_subset = get_cells_per_subset_2d_test(points,subset_bounds,adjacency_matrix,num_row,num_col)  
  #Building the graphs.
  graphs = bam.build_graphs(adjacency_matrix,num_row,num_col,num_angles)
  #Weighting the graphs with the preliminary info of the cells per subset and boundary cells per subset. This will also return the time to solve each subset.
  
  graphs,time_to_solve = add_edge_cost(graphs,subset_bounds,cells_per_subset,bdy_cells_per_subset,machine_params,num_row,num_col,False)
  graphs= pipeline_offset(graphs,num_angles,time_to_solve)
  #Making the edges universal.
  graphs = make_edges_universal(graphs)
  
  #Adding delay weighting.
  graphs = add_conflict_weights(graphs,time_to_solve,num_angles,unweighted)
  solve_times,max_time = compute_solve_time(graphs)
  end = time.time()
  max_time *= 100
  print(max_time,end-start)
  return max_time

def optimized_tts_3d(params,f,global_x_min,global_x_max,global_y_min,global_y_max,global_z_min,global_z_max,num_row,num_col,num_plane,machine_params,num_angles,Am,unweighted,test):
    
  x_cuts,y_cuts,z_cuts = unpack_parameters_3d(params,global_x_min,global_x_max,global_y_min,global_y_max,global_z_min,global_z_max,num_col,num_row,num_plane)
  #Building the subset boundaries.
  subset_bounds = b3a.build_3d_global_subset_boundaries(num_col-1,num_row-1,num_plane-1,x_cuts,y_cuts,z_cuts)
  #Getting mesh information.
  cells_per_subset, bdy_cells_per_subset = get_cells_per_subset_3d(f,subset_bounds)
  #Building the adjacency matrix.
  adjacency_matrix = b3a.build_adjacency_matrix(x_cuts,y_cuts,z_cuts,num_row,num_col,num_plane)
  #Building the graphs.
  graphs = b3a.build_graphs(adjacency_matrix,num_row,num_col,num_plane,num_angles)
  #Weighting the graphs based on cells per subset and boundary cells per subset.
  graphs,time_to_solve = add_edge_cost_3d(graphs,subset_bounds,cells_per_subset,bdy_cells_per_subset,machine_params,num_row,num_col,num_plane,Am,test)
  #Adjusting the graphs for multiple angles per octant.
  graphs= pipeline_offset(graphs,num_angles,time_to_solve)
  #Making the edges universal.
  graphs = make_edges_universal(graphs)
  #Adding delay weighting.
  graphs = add_conflict_weights(graphs,time_to_solve,num_angles,unweighted)
  solve_times,max_time = compute_solve_time(graphs)
  print(max_time) 
  return max_time

def optimized_tts_3d_numerical(params,points,global_x_min,global_x_max,global_y_min,global_y_max,global_z_min,global_z_max,num_row,num_col,num_plane,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted,test):
  start = time.time() 
  machine_params = (t_u,upc,upbc,t_comm,latency,m_l)
  
  x_cuts,y_cuts,z_cuts = unpack_parameters_3d(params,global_x_min,global_x_max,global_y_min,global_y_max,global_z_min,global_z_max,num_col,num_row,num_plane)
  x_cuts,y_cuts,z_cuts = tweak_parameters_3d(x_cuts,y_cuts,z_cuts,global_x_min,global_x_max,global_y_min,global_y_max,global_z_min,global_z_max,num_col,num_row,num_plane)
  print(z_cuts,x_cuts,y_cuts)
  #Building the subset boundaries.
  subset_bounds = b3a.build_3d_global_subset_boundaries(num_col-1,num_row-1,num_plane-1,x_cuts,y_cuts,z_cuts)
  #Getting mesh information.
  cells_per_subset, bdy_cells_per_subset = get_cells_per_subset_3d_numerical_test(points,subset_bounds)
  #Building the adjacency matrix.
  adjacency_matrix = b3a.build_adjacency_matrix(x_cuts,y_cuts,z_cuts,num_row,num_col,num_plane)
  #Building the graphs.
  graphs = b3a.build_graphs(adjacency_matrix,num_row,num_col,num_plane,num_angles)
  #Weighting the graphs based on cells per subset and boundary cells per subset.
  graphs,time_to_solve = add_edge_cost_3d(graphs,subset_bounds,cells_per_subset,bdy_cells_per_subset,machine_params,num_row,num_col,num_plane,test)
  #Adjusting the graphs for multiple angles per octant.
  graphs= pipeline_offset(graphs,num_angles,time_to_solve)
  #Making the edges universal.
  graphs = make_edges_universal(graphs)
  #Adding delay weighting.
  graphs = add_conflict_weights(graphs,time_to_solve,num_angles,unweighted)
  solve_times,max_time = compute_solve_time(graphs)
  end = time.time()
  print(max_time, end-start)
  
  return max_time
