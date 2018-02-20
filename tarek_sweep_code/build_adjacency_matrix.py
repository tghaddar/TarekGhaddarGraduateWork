#This builds the adjacency matrix for all subsets. ycuts are stored by column. 
def build_adjacency(global_bounds,n_x, n_y, ycuts):
  
  #The number of subsets in our domain.
  num_subsets = len(global_bounds)
  
  adjacency_matrix = []
  
  for s in range(0, num_subsets):
    #The neighbors of this subset.
    neighbors = []
    #The bounds of this subset.
    ymin = global_bounds[s][2]
    ymax = global_bounds[s][3]

    
    i_val = int(s/(n_y+1))
    
    #The number of interior ycuts in each column
    n_y = len(ycuts[0]) - 2
    
    #The number of columns in our domain.
    numcol = n_x+1
    #The number of rows in our domain.
    numrow = n_y + 1
    
    #We are only adding right and left neighborsif our domain is more than one subset wide
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
    
    adjacency_matrix.append(neighbors)        
    
  
  return adjacency_matrix

