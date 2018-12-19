import numpy as np
from utilities import get_ij
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
def flip_adjacency(adjacency_matrix,numrow,numcol):
  
  num_subsets = numrow*numcol
  adjacency_flip = np.zeros((num_subsets,num_subsets))
  #Maps old ids to new ones.
  id_map = {}
  
  for s in range(0,num_subsets):
    #Getting the i,j indices of the subset.
    i,j = get_ij(s,numrow,numcol)
    #The maximum subset in this column when ordered conventionally.
    max_subset = i*numrow + (numrow-1)
    #The new subset id in the flipped ordering.
    new_ss_id = max_subset - j
    #Recording the new id.
    id_map[s] = new_ss_id
    #Pulling the neighbor info when subsets are conventionally ordered.
    original_neighbors = adjacency_matrix[s]
    #Getting the indices that are neighbors.
    indices = np.where(original_neighbors == 1)[0]
    #Intializing the new neighbors.
    new_neighbors = np.zeros(num_subsets)
    #Looping over indices
    for ind in range(0,len(indices)):
      #Original nieghbor id
      neighbor_id = indices[ind]
      #Getting the i,j indices of this neighbor.
      i_ind,j_ind = get_ij(neighbor_id,numrow,numcol)
      #Getting the maximum subset of this neighbor when ordered conventionally.
      max_subset_neighbor = i_ind*numrow + (numrow-1)
      #The new subset id of this neighbor in the flipped ordering.
      new_neighbor_id = int(max_subset_neighbor - j_ind)
      #Adding this neighbor into the new neighbors.
      new_neighbors[new_neighbor_id] = 1 
    
    #Adding the neighbors to the flipped adjacency matrix when ordered differently.
    adjacency_flip[new_ss_id] = new_neighbors
      
  return adjacency_flip,id_map
    
    
      