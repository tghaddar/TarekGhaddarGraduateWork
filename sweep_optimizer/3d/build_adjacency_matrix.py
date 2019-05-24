import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings
from copy import deepcopy
from flip_adjacency_2d import flip_adjacency
warnings.filterwarnings("ignore", category=DeprecationWarning)

#This builds the adjacency matrix for all subsets. ycuts are stored by column. 
def build_adjacency(global_bounds,n_x, n_y, ycuts):
  
  #The number of subsets in our domain.
  num_subsets = len(global_bounds)
  
  adjacency_list = []
  
  for s in range(0, num_subsets):
    #The neighbors of this subset.
    neighbors = []
    #The bounds of this subset.
    ymin = global_bounds[s][2]
    ymax = global_bounds[s][3]

    
    i_val = int(s/(n_y+1))
    j_val = int(s - i_val*(n_y + 1))
    
    #The number of interior ycuts in each column
    n_y = len(ycuts[0]) - 2
    
    #The number of columns in our domain.
    numcol = n_x+1
    #The number of rows in our domain.
    numrow = n_y + 1
    
    #We only add top or bottom neighbors if our domain is more than one subset tall
    if (numrow != 1):
      if (j_val == 0):
        neighbors.append(s+1)
      elif (j_val == numrow - 1):
        neighbors.append(s-1)
      else:
        neighbors.append(s+1)
        neighbors.append(s-1)
    
    #We are only adding right and left neighbors if our domain is more than one subset wide
    if (numcol != 1):
      #If we're in any column but the last one, we look to our right for potential neighbors.
      if (i_val < numcol - 1):
        #The ycut lines in the column to the right.
        right_column_y_cuts = ycuts[i_val+1]
        #Looping over all cuts but the top one.
        for j in range (0, n_y+1):
          #Grabbing the enclosure of the potential neighboring subset.
          cut = right_column_y_cuts[j]
          next_cut = right_column_y_cuts[j+1]
          if ( (next_cut == ymin) or (cut == ymax) ):
            continue
          elif (ymin < next_cut and ymax > cut):
            neighbors.append((i_val+1)*numrow + j)
      #If we're in any column but the first column, we look to our left neighbor.
      if (i_val > 0):
        #The ycut lines in the column to the left.
        left_column_y_cuts = ycuts[i_val - 1]
        for j in range(0, n_y + 1):
          cut = left_column_y_cuts[j]
          next_cut = left_column_y_cuts[j+1]
          if ( (next_cut == ymin) or (cut == ymax) ):
            continue;
          if ( ymin < next_cut and ymax > cut ):
            neighbors.append((i_val-1)*numrow + j )
    
    adjacency_list.append(neighbors)        

  #The adjacency matrix in matrix form instead of in sparse list form.
  adjacency_matrix = np.zeros((num_subsets,num_subsets))
  
  for i in range(0,num_subsets):
    
    neighbors = adjacency_list[i]
    for j in range(0,len(neighbors)):
      
      adjacency_matrix[i][neighbors[j]] = 1
  
  return adjacency_matrix

def build_graphs(adjacency_matrix,num_row,num_col,num_angle):

  #Getting the upper triangular portion of the adjacency_matrix
  adjacency_matrix_0 = np.triu(adjacency_matrix)
  #Time to build the graph
  G = nx.DiGraph(adjacency_matrix_0)
  #Adding the -1 node.
  end_node = [x for x in G.nodes() if G.out_degree(x) == 0][0]
  G.add_node(-1)
  G.add_edge(end_node,-1)
  
  #Test what lower triangular looks like
  adjacency_matrix_3 = np.tril(adjacency_matrix)
  G3 = nx.DiGraph(adjacency_matrix_3)
   #Adding the -1 node.
  end_node = [x for x in G3.nodes() if G3.out_degree(x) == 0][0]
  G3.add_node(-1)
  G3.add_edge(end_node,-1)
  
  
  #To get the top left and bottom right quadrants, we have to reverse our ordering by column.
  adjacency_flip,id_map = flip_adjacency(adjacency_matrix,num_row,num_col)
  adjacency_matrix_1 = np.triu(adjacency_flip)
  
  G1 = nx.DiGraph(adjacency_matrix_1)
  G1 = nx.relabel_nodes(G1,id_map,copy=True)
   #Adding the -1 node.
  end_node = [x for x in G1.nodes() if G1.out_degree(x) == 0][0]
  G1.add_node(-1)
  G1.add_edge(end_node,-1)
  
  #Bottom right quadrant.
  adjacency_matrix_2 = np.tril(adjacency_flip)
  G2 = nx.DiGraph(adjacency_matrix_2)
  G2 = nx.relabel_nodes(G2,id_map,copy=True)
   #Adding the -1 node.
  end_node = [x for x in G2.nodes() if G2.out_degree(x) == 0][0]
  G2.add_node(-1)
  G2.add_edge(end_node,-1)
  
  graphs = [G,G1,G2,G3]
  for angle in range(0,num_angle-1):
    graphs.append(deepcopy(G))
    graphs.append(deepcopy(G1))
    graphs.append(deepcopy(G2))
    graphs.append(deepcopy(G3))

  return graphs