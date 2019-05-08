from mesh_processor import analytical_mesh_integration_2d,create_2d_cuts,get_cells_per_subset_2d,create_2d_cut_suite
from build_global_subset_boundaries import build_global_subset_boundaries
from sweep_solver import time_to_solution,add_edge_cost,make_edges_universal,add_conflict_weights,get_y_cuts
from build_adjacency_matrix import build_graphs,build_adjacency


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

#Number of rows and columns.
numrow = 2
numcol = 2

#Global boundaries.
global_xmin = 0.0
global_xmax = 10.0
global_ymin = 0.0
global_ymax = 10.0

#The subset boundaries.
x_cuts,y_cuts = create_2d_cuts(global_xmin,global_xmax,numcol,global_ymin,global_ymax,numrow)

all_x_cuts,all_y_cuts = create_2d_cut_suite(global_xmin,global_xmax,numcol,global_ymin,global_ymax,numrow)

#The subset_boundaries.
subset_boundaries = build_global_subset_boundaries(numcol-1,numrow-1,x_cuts,y_cuts)

f = lambda x,y: x+y

#cells_per_subset, bdy_cells_per_subset = get_cells_per_subset_2d(f,subset_boundaries)
#
#ycuts = get_y_cuts(subset_boundaries,numrow,numcol)
##Building the adjacency matrix.
#adjacency_matrix = build_adjacency(subset_boundaries,numcol-1,numrow-1,ycuts)
##Building the graphs.
#graphs = build_graphs(adjacency_matrix,numrow,numcol)
#
##Weighting the graphs with the preliminary info of the cells per subset and boundary cells per subset. This will also return the time to solve each subset.
#graphs,time_to_solve = add_edge_cost(graphs,subset_boundaries,cells_per_subset,bdy_cells_per_subset,machine_params,numrow,numcol)
#graphs = make_edges_universal(graphs)
#graphs = add_conflict_weights(graphs,time_to_solve)


max_time = time_to_solution(f,subset_boundaries,machine_params,numcol,numrow)
#plot_graphs(graphs,0)


