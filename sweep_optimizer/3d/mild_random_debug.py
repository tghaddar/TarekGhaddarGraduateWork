import numpy as np
import matplotlib.pyplot as plt
import build_adjacency_matrix as bam
from sweep_solver import add_edge_cost,pipeline_offset,make_edges_universal,add_conflict_weights,compute_solve_time
from sweep_solver import plot_subset_boundaries_2d, plot_graphs
from build_global_subset_boundaries import build_global_subset_boundaries
plt.close("all")

x_cuts = np.genfromtxt("x_cuts_3.csv",delimiter=',')
y_cuts = np.genfromtxt("y_cuts_3.csv",delimiter=',')

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

num_row = 3
num_col = 3
num_subsets = num_row*num_col
num_angles = 1
test = True

#Building subset boundaries.
subset_bounds = build_global_subset_boundaries(num_col-1,num_row-1,x_cuts,y_cuts)
plot_subset_boundaries_2d(subset_bounds,num_subsets)
#Getting mesh information.
#Dummy values for the purpose of this test case. 
cells_per_subset = [1 for n in range(0,num_subsets)]
bdy_cells_per_subset = [[1,1] for n in range(0,num_subsets)]
#Building the adjacency matrix.
adjacency_matrix = bam.build_adjacency(subset_bounds,num_col-1,num_row-1,y_cuts)
#Building the graphs.
graphs = bam.build_graphs(adjacency_matrix,num_row,num_col,num_angles)
#Weighting the graphs with the preliminary info of the cells per subset and boundary cells per subset. This will also return the time to solve each subset.
graphs,time_to_solve = add_edge_cost(graphs,subset_bounds,cells_per_subset,bdy_cells_per_subset,machine_params,num_row,num_col,test)
graphs = pipeline_offset(graphs,num_angles,time_to_solve)
#Making the edges universal.
graphs = make_edges_universal(graphs)
plot_graphs(graphs,0,0,num_angles)
#Adding delay weighting.
graphs = add_conflict_weights(graphs,time_to_solve,num_angles)
solve_times,max_time = compute_solve_time(graphs)


