import numpy as np
import matplotlib.pyplot as plt

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


  plt.figure(1)
  subset_centers = []
  for i in range(0,num_subsets):
    subset_boundary = global_bounds[i]
    xmin = subset_boundary[0]
    xmax = subset_boundary[1]
    ymin = subset_boundary[2]
    ymax = subset_boundary[3]
  
    center_x = (xmin+xmax)/2
    center_y = (ymin+ymax)/2
  
    subset_centers.append([center_x, center_y])
  
    x = [xmin, xmax, xmax, xmin, xmin]
    y = [ymin, ymin, ymax, ymax, ymin]
  
    plt.plot(x,y,'b')
  
  
  for i in range (0, num_subsets):
  
    neighbors = adjacency_list[i]
  
    for j in range(0, len(neighbors)):
      n = neighbors[j]
  
      x = [subset_centers[i][0], subset_centers[n][0]]
      y = [subset_centers[i][1], subset_centers[n][1]]
      plt.plot(x,y,'r-o')
  
  plt.savefig('adjacency_matrix.pdf')


  #The adjacency matrix in matrix form instead of in sparse list form.
  adjacency_matrix = np.zeros((num_subsets,num_subsets))
  
  for i in range(0,num_subsets):
    
    neighbors = adjacency_list[i]
    for j in range(0,len(neighbors)):
      
      adjacency_matrix[i][neighbors[j]] = 1
  
  return adjacency_matrix

