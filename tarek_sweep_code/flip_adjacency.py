def flip_adjacency(adjacency_matrix,numrow,numcol):
    
  for col in range(0,numcol):
    
    #The minimum subset in this column when ordered conventionally.
    min_subset = col*numrow
    #The maximum subset in this column when ordered conventionally.
    max_subset = col*numrow + (numrow-1)
    
    
    
  