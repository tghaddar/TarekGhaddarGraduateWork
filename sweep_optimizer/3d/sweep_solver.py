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
      
#Get the sum of weights to a path.
def get_weight_sum(graph,path,node):
  weight_sum = 0.0
  node_index = -1
  try:
    node_index = path.index(node)
  except:
    #If this node is not this path, we return a very large sum.
    return 1e8
  
  weight_sum = 0.0
  for j in range(0,node_index):
    node1 = path[j]
    node2 = path[j+1]
    weight_sum += graph[node1][node2]['weight']
  
  return weight_sum

#Returns the depth of graph remaining given a heavy path.
def get_DOG(graph,path,node):

  weight_sum = 0.0
  node_index = path.index(node)
  for n in range(node_index,len(path)-1):
    node1 = path[n]
    node2 = path[n+1]
    weight_sum += graph[node1][node2]['weight']
  
  return weight_sum

#Sorts path indices based on priority octants.
def sort_priority(path_indices,paths,dogs,dogs_remaining,graph_indices):
  #The true octant priorities.
  true_priorities = [0,4,1,5,2,6,3,7]
  original_indices = copy(path_indices)
  original_paths = copy(paths)
  original_dogs = copy(dogs)
  original_dogs_remaining = copy(dogs_remaining)
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
    
  for i in range(0,len(paths)):
    old_index = index_map[i]
    paths[i] = original_paths[old_index]
    dogs[i] = original_dogs[old_index]
    dogs_remaining[i] = original_dogs_remaining[old_index]
    path_indices[i]= original_indices[old_index]
    
  return path_indices,paths,dogs,dogs_remaining,graph_indices


#We are figuring out which is the path with priority based on which octant it belongs to.
def get_priority_octant(path1,path2):
  
  #Boolean checking if omega_x > 0 for both paths.
  path1_ox = (path1 == 0 or path1 == 1 or path1 == 4 or path1 == 5)
  path2_ox = (path2 == 0 or path2 == 1 or path2 == 4 or path2 == 5)
  #Boolean checking if omega_y > 0 for both paths.
  path1_oy = (path1 == 0 or path1 == 2 or path1 == 4 or path1 == 6)
  path2_oy = (path2 == 0 or path2 == 2 or path2 == 4 or path2 == 6)
  #Boolean checking if omega_z > 0 for both paths
  path1_oz = (path1 == 0 or path1 == 1 or path1 == 2 or path1 == 3)
  path2_oz = (path2 == 0 or path2 == 1 or path2 == 2 or path2 == 3)
  
  #Checking who has omega_x > 0.
  if (path1_ox == True and path2_ox == False):
    return "primary"
  elif (path1_ox == False and path2_ox == True):
    return "secondary"
  #If both have omega_x > 0 or omega_x < 0 then we check omega_y
  else:
    if (path1_oy == True and path2_oy == False):
      return "primary"
    elif (path1_oy == False and path2_oy == True):
      return "secondary"
    #If both omega_x and omega_y > 0 for both paths then check omega_z
    else:
      if (path1_oz == True and path2_oz == False):
        return "primary"
      elif (path1_oz == False and path2_oz == True):
        return "secondary"
      else:
        return "problem"

#Takes simple paths for a graph and dumps them out.
def print_simple_paths(path_gen):
  for p in path_gen:
    print(p)

def convert_generator(simple_paths):
  
  new_simple_paths = []
  for i in range(0,len(simple_paths)):
    newpath = simple_paths[i]
    newpath = list(newpath)
    new_simple_paths.append(newpath)
  
  return new_simple_paths

def add_conflict_weights(graphs,all_simple_paths,latency,cell_dist,num_row,num_col,num_plane):
  
  num_nodes = graphs[0].number_of_nodes()
  
  
  #A boolean value that checks if we are perfectly balanced.
  is_perfect = all(x == cell_dist[0] for x in cell_dist)
  #We use a slightly different conflict resolution for perfectly balanced problems.
  if (is_perfect):
    #The static delay added in conflicts is just the wait to solve and communicate.
    delay = copy(graphs[0][0][1]['weight'])
    #Length of each individual path.
    path_length = 0
    #Converting to a list of lists rather than a list of generators for iterative purposes.
    all_simple_paths = convert_generator(all_simple_paths)
    
    all_octant_paths = []
    all_graph_correspondence = []
    for n in range(0,num_nodes):
      octant_paths = []
      graph_correspondence = []
      #Looping through all paths for all graphs in order to determine a path for each octant that contains this node.
      for p in range(0,len(graphs)):
        #Simple paths for this octant.
        perf_paths = copy(all_simple_paths)
        this_simple_paths = perf_paths[p]
        for path in this_simple_paths:
          path_length = len(path)
          if (n in path):
            octant_paths.append(path)
            graph_correspondence.append(p)
            #break
#      We use these octant paths to figure out if we have any conflicts.    
#      For this node, find out which paths will conflict. We know the path with the node as it's originator won't conflict ever.
#      Looping over to see which nodes potentially conflict on these paths.
      for c in range(1,path_length-1):
        conflicting_paths = []
        conflicting_indices = []
        #DOG remaining for all paths.
        conflicting_dog_remaining = []
        #DOG for all conflicting paths
        conflicting_dog = []
        conflicting_graph_indices = []
        for p in range(0,len(octant_paths)):
          index = octant_paths[p].index(n)
          if (index == c):
            conflicting_paths.append(octant_paths[p])
            conflicting_indices.append(p)
            conflicting_graph_indices.append(graph_correspondence[p])
            dog_remaining = get_DOG(graphs[graph_correspondence[p]],octant_paths[p],n)
            dog = get_weight_sum(graphs[graph_correspondence[p]],octant_paths[p],n)
            conflicting_dog_remaining.append(dog_remaining)
            conflicting_dog.append(dog)
            
        
        #Loop over all conflicting paths and remove winners one by one until all delays are addressed. 
        p = 0
        while p < len(conflicting_paths):
          if (len(conflicting_paths) > 1):
            #Check if all nodes are ready to solve.
            tied = all(x == conflicting_dog[0] for x in conflicting_dog)
            if (tied):
              #Check if all nodes have the samef  depth of graph remaining.
              dog_remaining_tie = all(x == conflicting_dog_remaining[0] for x in conflicting_dog_remaining)
              #Check if the depth of graph remaining is equivalent.
              if (dog_remaining_tie):
                #Resorting everything based on priority octant rules.
                conflicting_indices,conflicting_paths,conflicting_dog,conflicting_dog_remaining,conflicting_graph_indices = sort_priority(conflicting_indices,conflicting_paths,conflicting_dog,conflicting_dog_remaining,conflicting_graph_indices)
                
                del conflicting_indices[0]
                del conflicting_paths[0]
                del conflicting_dog[0]
                del conflicting_dog_remaining[0]
                del conflicting_graph_indices[0]
                for i in range(0,len(conflicting_indices)):
                  #index = conflicting_indices[i]
                  graph_index = conflicting_graph_indices[i]
                  losing_next_node = conflicting_paths[i][c+1]
                  graphs[graph_index][n][losing_next_node]['weight'] += delay

                p -= 1
              else:
                #The one with the maximum dog_remaining wins.
                max_dog_remaining = max(conflicting_dog_remaining)
                max_dog_remaining_index = conflicting_dog_remaining.index(max_dog_remaining)


                #Removing this path from conflicting paths since it gets priority and starts solving.
                del conflicting_paths[max_dog_remaining_index]
                del conflicting_indices[max_dog_remaining_index]
                del conflicting_dog[max_dog_remaining_index]
                del conflicting_dog_remaining[max_dog_remaining_index]
                del conflicting_graph_indices[max_dog_remaining_index]
                for i in range(0,len(conflicting_indices)):
                  index = conflicting_indices[i]
                  graph_index = conflicting_graph_indices[i]
                  losing_next_node = conflicting_paths[i][c+1]
                  graphs[graph_index][n][losing_next_node]['weight'] += delay
                p -=1
                
            else:
              #Get minimum dog (corresponds to the path that reaches first).
              min_dog = min(conflicting_dog)
              min_dog_index = conflicting_dog.index(min_dog)
              
              del conflicting_paths[min_dog_index]
              del conflicting_indices[min_dog_index]
              del conflicting_dog[min_dog_index]
              del conflicting_dog_remaining[min_dog_index]
              del conflicting_graph_indices[min_dog_index]
              
              for i in range(0,len(conflicting_indices)):
                index = conflicting_indices[i]
                graph_index = conflicting_graph_indices[i]
                losing_next_node = conflicting_paths[i][c+1]
                graphs[graph_index][n][losing_next_node]['weight'] += delay
              
              p -= 1
          
          p += 1
      all_octant_paths.append(octant_paths)
      all_graph_correspondence.append(graph_correspondence)
  
    #Picking up delays we missed.
    count = 0
    while(count < 1):
      for n in range(0,num_nodes):
        octant_paths = all_octant_paths[n]
        graph_correspondence = all_graph_correspondence[n]
        for p in range(0,len(octant_paths)):
          path1 = octant_paths[p]
          graph_index1 = graph_correspondence[p]
          graph1 = graphs[graph_index1]
          dog1 = get_weight_sum(graph1,path1,n)
          dog_remaining1 = get_DOG(graph1,path1,n)
          
          for pi in range(p+1,len(octant_paths)):
            path2 = octant_paths[pi]
            graph_index2 = graph_correspondence[pi]
            graph2 = graphs[graph_index2]
            if (path1[0] == path2[0]):
              continue
            if (path1[-1] == n or path2[-1] == n):
              continue
            dog2 = get_weight_sum(graph2,path2,n)
            dog_remaining2 = get_DOG(graph2,path2,n)
            
            dog_remaining_tied = (dog_remaining1 == dog_remaining2)
            graph_indices = [graph_index1,graph_index2]
            indices = [p,pi]
            paths = [path1,path2]
            dogs_remaining = [dog_remaining1,dog_remaining2]
            dogs = [dog1,dog2]
            #Checking if the two graphs have the same DOG.
            tied = (dog1 == dog2)
            if (tied):
              
              #Checking to see if the two graphs have the same DOG remaining.
              if (dog_remaining_tied):
                
                indices,paths,dogs,dogs_remaining,graph_indices = sort_priority(indices,paths,dogs,dogs_remaining,graph_indices)
                for i in range(1,len(indices)):
                  #index = paths[i].index(n)
                  graph_index = graph_indices[i]
                  next_node_index = paths[i].index(n)+1
                  losing_next_node = paths[i][next_node_index]
                  graphs[graph_index][n][losing_next_node]['weight'] += delay
              else:
                max_dog_remaining = max(dogs_remaining)
                max_dog_remaining_index = dogs_remaining.index(max_dog_remaining)
                
                #Removing this path from conflicting paths since it gets priority and starts solving.
                del paths[max_dog_remaining_index]
                del indices[max_dog_remaining_index]
                del dogs[max_dog_remaining_index]
                del dogs_remaining[max_dog_remaining_index]
                del graph_indices[max_dog_remaining_index]
                
                #index = paths[0].index(n)
                graph_index = graph_indices[0]
                next_node_index = paths[0].index(n)+1
                losing_next_node = paths[0][next_node_index]
                graphs[graph_index][n][losing_next_node]['weight'] += delay
            else:
              min_dog = min(dogs)
              min_dog_index = dogs.index(min_dog)
              
              if (abs(dog1 - dog2) >= delay):
                continue
              else:
                del paths[min_dog_index]
                del indices[min_dog_index]
                del dogs[min_dog_index]
                del dogs_remaining[min_dog_index]
                del graph_indices[min_dog_index]
                
                #index = paths[0].index(n)
                graph_index = graph_indices
                next_node_index = paths[0].index(n)+1
                losing_next_node = paths[0][next_node_index]
                graphs[graph_index][n][losing_next_node]['weight'] += delay
      count += 1
  
  else:
  
    for p in range(0,len(all_simple_paths)):
      current_path = all_simple_paths[p]
      heavy_path,path_weight = get_heaviest_path(graphs[p],current_path)
      all_simple_paths[p] = heavy_path
    
    
    for n in range(0,num_nodes):
      fastest_path,weight_sum = get_fastest_path(graphs,all_simple_paths,n)
        
      primary_graph = graphs[fastest_path]
      primary_path = all_simple_paths[fastest_path]
      primary_index = primary_path.index(n)
      
      #Looping through remaining path to add potential delays for this node.
      for p in range(0,len(all_simple_paths)):
        secondary_path = all_simple_paths[p]
        secondary_graph = graphs[p]
        if p == fastest_path:
          continue
        #Check if this node exists in the secondary path.
        secondary_index = -1
        try:
          secondary_index = secondary_path.index(n)
        except:
          continue
        
        weight_sum_secondary = get_weight_sum(graphs[p],all_simple_paths[p],n)
        delay = weight_sum_secondary - weight_sum
        time_to_solve = primary_graph[n][primary_path[primary_index+1]]['weight']
        delay = time_to_solve - delay
        #If two graphs reach each other at the same time, we have to resort to depth of graph.
        if (isclose(delay,0.0,rel_tol=1e-4*latency)):
          print("same time")
          dog_primary = get_DOG(primary_graph,primary_path,n)
          dog_secondary = get_DOG(secondary_graph,secondary_path,n)
          if (dog_primary > dog_secondary):
            next_node = secondary_path[secondary_index+1]
            secondary_graph[n][next_node]['weight'] += time_to_solve
          elif (dog_secondary > dog_primary):
            next_node = primary_path[primary_index+1]
            primary_graph[n][next_node]['weight'] += time_to_solve
          else:
            #Need to figure out which path has priority based on octant
            which_path = get_priority_octant(fastest_path,p)
            if (which_path == "primary"):
              next_node = secondary_path[secondary_index+1]
              secondary_graph[n][next_node]['weight'] += delay
            elif(which_path == "secondary"):
              next_node = primary_path[primary_index+1]
              primary_graph[n][next_node]['weight'] += time_to_solve
            else:
              raise("Error, we need a primary")
        elif (delay > 0):
          #Add this delay to the current node's solve time in the secondary graph.
          next_node = secondary_path[secondary_index+1]
          secondary_graph[n][next_node]['weight'] += delay
        
      
  return graphs
  
#Gets the path that gets fastest to a node. Assumes that each graph has its heaviest path listed.
def get_fastest_path(graphs,paths,node):
  
  check_paths = copy(paths)
  #Checks if the node is in the path.
  weight_sum = 1e8
  fastest_path = 0
  index = 0
  for path in check_paths:
    graph = graphs[index]
    node_index = -1
    try:
      node_index = path.index(node)
    except:
      index += 1
      continue
    
    weight_sum_path = 0.0
    for j in range(0,node_index):
      node1 = path[j]
      node2 = path[j+1]
      weight_sum_path += graph[node1][node2]['weight']
    
    #If this path is fastest (smallest weight), we update our fastest_path variable.
    if (weight_sum_path < weight_sum):
      weight_sum = weight_sum_path
      fastest_path = index
    
    index += 1
  
  return fastest_path,weight_sum


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
#    heaviest_path = 0.0
#    for path in paths:
#      path_weight = sum_weights_of_path(graph,path)
#      if path_weight > heaviest_path:
#        heaviest_path = path_weight
#        index = path
#    
#    heaviest_paths.append(index)
    heaviest_path,path_weight = get_heaviest_path(graph,paths)
    heaviest_paths.append(heaviest_path)
    time_graph = path_weight + t_u*upc*cells_per_subset[end_node]
    all_graph_time[ig] = time_graph 
  
  time = np.average(all_graph_time)
  return all_graph_time,time,heaviest_paths
    