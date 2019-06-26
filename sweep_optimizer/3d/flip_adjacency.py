import numpy as np
from utilities import get_ijk
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
def flip_adjacency_1(adjacency_matrix,numrow,numcol,num_plane):
  
  num_subsets = numrow*numcol*num_plane
  adjacency_flip = np.zeros((num_subsets,num_subsets))
  #Maps old ids to new ones.
  id_map = {}
  num_subsets_2d = numrow*numcol
  
  #Flipping by layer.
  for s in range(0,num_subsets):
    #Getting the i,j indices of the subset.
    i,j,k = get_ijk(s,numrow,numcol,num_plane)
    #The maximum subset in this column when ordered conventionally.
    max_subset = k*num_subsets_2d  + i*numrow + (numrow-1)
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
      i_ind,j_ind,k_ind = get_ijk(neighbor_id,numrow,numcol,num_plane)
      #Getting the maximum subset of this neighbor when ordered conventionally.
      max_subset_neighbor = k_ind*num_subsets_2d + i_ind*numrow + (numrow-1)
      #The new subset id of this neighbor in the flipped ordering.
      new_neighbor_id = int(max_subset_neighbor - j_ind)
      #Adding this neighbor into the new neighbors.
      new_neighbors[new_neighbor_id] = 1 
    
    #Adding the neighbors to the flipped adjacency matrix when ordered differently.
    adjacency_flip[new_ss_id] = new_neighbors
      
  return adjacency_flip,id_map
    

def flip_adjacency_2(adjacency_matrix,numrow,numcol,num_plane):
  
  num_subsets = numrow*numcol*num_plane
  adjacency_flip = np.zeros((num_subsets,num_subsets))
  #Maps old ids to new ones.
  id_map = {}
  num_subsets_2d = numrow*numcol
  
  #Flipping by layer.
  for s in range(0,num_subsets):
    #Getting the i,j indices of the subset.
    i,j,k = get_ijk(s,numrow,numcol,num_plane)
    #Getting the new subset id if octant 2 starts with subset 0.
    new_i = numcol - i - 1
    new_ss_id = k*num_subsets_2d + (new_i*numrow+j) 
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
      i_ind,j_ind,k_ind = get_ijk(neighbor_id,numrow,numcol,num_plane)
      #Getting the new i_ind
      new_i_ind = numcol - i_ind - 1
      new_neighbor_id = k_ind*num_subsets_2d + (new_i_ind*numrow+j_ind) 
      #Adding this neighbor into the new neighbors.
      new_neighbors[new_neighbor_id] = 1
    
    #Adding the neighbors to the flipped adjacency matrix when ordered differently.
    adjacency_flip[new_ss_id] = new_neighbors
  
  return adjacency_flip,id_map


def flip_adjacency_3(adjacency_matrix,numrow,numcol,num_plane):
  
  num_subsets = numrow*numcol*num_plane
  adjacency_flip = np.zeros((num_subsets,num_subsets))
  #Maps old ids to new ones.
  id_map = {}
  num_subsets_2d = numrow*numcol
  #Flipping by layer.
  for s in range(0,num_subsets):
    #Getting the i,j indices of the subset.
    i,j,k = get_ijk(s,numrow,numcol,num_plane)
    new_i = numcol - i - 1
    new_j = numrow - j - 1
    
    new_ss_id = k*num_subsets_2d + (new_i*numrow+new_j)
    #Recording the new id.
    id_map[s] = new_ss_id
    #Pulling the neighbor info when subsets are conventionally ordered.
    original_neighbors = adjacency_matrix[s]
    #Getting the indices that are neighbors.
    indices = np.where(original_neighbors == 1)[0]
    #Intializing the new neighbors.
    new_neighbors = np.zeros(num_subsets)
    
    for ind in range(0,len(indices)):
      #Original nieghbor id
      neighbor_id = indices[ind]
      #Getting the i,j indices of this neighbor.
      i_ind,j_ind,k_ind = get_ijk(neighbor_id,numrow,numcol,num_plane)
      #Getting the new i_ind
      new_i_ind = numcol - i_ind - 1
      new_j_ind = numrow - j_ind - 1
      new_neighbor_id = k_ind*num_subsets_2d + (new_i_ind*numrow+new_j_ind) 
      #Adding this neighbor into the new neighbors.
      new_neighbors[new_neighbor_id] = 1
    
    #Adding the neighbors to the flipped adjacency matrix when ordered differently.
    adjacency_flip[new_ss_id] = new_neighbors
  
  return adjacency_flip,id_map
    
    
    