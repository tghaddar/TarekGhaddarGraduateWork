from build_3d_adjacency import build_3d_global_subset_boundaries
from build_3d_adjacency import build_adjacency_matrix
from build_3d_adjacency import build_graphs
from sweep_solver import make_edges_universal
from sweep_solver import add_conflict_weights,pipeline_offset
from sweep_solver import compute_solve_time,add_edge_cost
from sweep_solver import get_max_incoming_weight,optimized_tts_3d
from mesh_processor import create_3d_cuts
from optimizer import create_parameter_space_3d
import matplotlib.pyplot as plt 
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#The machine parameters.
#Communication time per double
t_comm = 4.47e-02
#The number of bytes to communicate per subset.
#The message latency time.
m_l = 1
latency = 4110.0e-02
#Solve time per unknown.
t_u = 450.0e-02
upc = 8.0
upbc = 4.0

unweighted = True
test = True

machine_params = (t_u,upc,upbc,t_comm,latency,m_l)

#Global boundaries.
global_x_min = 0.0
global_x_max = 10.0
global_y_min = 0.0
global_y_max = 10.0
global_z_min = 0.0
global_z_max = 10.0

num_row = 5
num_col = num_row
num_plane = num_row
num_subsets = num_row*num_col*num_plane
num_angles = 2

z_cuts,x_cuts,y_cuts = create_3d_cuts(global_x_min,global_x_max,num_col,global_y_min,global_y_max,num_row,global_z_min,global_z_max,num_plane)
    
subset_boundaries = build_3d_global_subset_boundaries(num_col-1,num_row-1,num_plane-1,x_cuts,y_cuts,z_cuts)

adjacency_matrix = build_adjacency_matrix(x_cuts,y_cuts,z_cuts,num_row,num_col,num_plane)

graphs = build_graphs(adjacency_matrix,num_row,num_col,num_plane,num_angles)
num_graphs = len(graphs)
#A list that stores the time to solve each node.
#Dummy values for the purpose of this test case. 
cells_per_subset = [1 for n in range(0,num_subsets)]
bdy_cells_per_subset = [[1,1] for n in range(0,num_subsets)]
#Adding the universal cost.
graphs,time_to_solve = add_edge_cost(graphs,subset_boundaries,cells_per_subset,bdy_cells_per_subset,machine_params,num_row,num_col,True)
graphs = pipeline_offset(graphs,num_angles,time_to_solve)

graphs = make_edges_universal(graphs)

graphs = add_conflict_weights(graphs,time_to_solve,num_angles,unweighted)
time_to_soln = compute_solve_time(graphs)[1]
print(time_to_soln)


f  = lambda x,y,z: 5
params = create_parameter_space_3d(x_cuts,y_cuts,z_cuts,num_row,num_col,num_plane)

optimized_time_to_soln = optimized_tts_3d(params,f,global_x_min,global_x_max,global_y_min,global_y_max,global_z_min,global_z_max,num_row,num_col,num_plane,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted,test)
print(optimized_time_to_soln)