from sweep_solver import get_DOG_remaining, get_DOG_remaining_unweighted,compute_solve_time
from sweep_solver import add_edge_cost,pipeline_offset,make_edges_universal,add_conflict_weights
from build_adjacency_matrix import build_adjacency,build_graphs
from build_global_subset_boundaries import build_global_subset_boundaries
import numpy as np


#The machine parameters.
#Communication time per double
t_comm = 4.47e-02
#The number of bytes to communicate per subset.
#The message latency time.
m_l = 1
latency = 4110.0e-02
#Solve time per unknown.
t_u = 450.0e-02
upc = 4.0
upbc = 2.0

machine_params = (t_u,upc,upbc,t_comm,latency,m_l)

#The 3 by 3 case.
x_cuts = list(np.genfromtxt("x_cuts_3.csv",delimiter=","))
y_cuts = np.genfromtxt("y_cuts_3.csv",delimiter=",")

num_angles = 1
num_space = 3
numrow = num_space
numcol = num_space
num_subsets = numrow*numcol

#The subset_boundaries.
subset_boundaries = build_global_subset_boundaries(numcol-1,numrow-1,x_cuts,y_cuts)
#Building the adjacency matrix.
adjacency_matrix = build_adjacency(subset_boundaries,numcol-1,numrow-1,y_cuts)
#Building the graphs
graphs = build_graphs(adjacency_matrix,numrow,numcol,num_angles)


num_graphs = len(graphs)
#Dummy values for the purpose of this test case. 
cells_per_subset = [1 for n in range(0,num_subsets)]
bdy_cells_per_subset = [[1,1] for n in range(0,num_subsets)]

#Adding the universal cost.
graphs,time_to_solve = add_edge_cost(graphs,subset_boundaries,cells_per_subset,bdy_cells_per_subset,machine_params,numrow,numcol,True)

graphs = pipeline_offset(graphs,num_angles,time_to_solve)
  
graphs = make_edges_universal(graphs)
  

graphs = add_conflict_weights(graphs,time_to_solve,num_angles,True)
print(compute_solve_time(graphs))