#This script generates data for 2D verification on the mild random layout.
from build_global_subset_boundaries import build_global_subset_boundaries
from build_adjacency_matrix import build_graphs,build_adjacency
from sweep_solver import add_edge_cost,pipeline_offset,make_edges_universal
from sweep_solver import add_conflict_weights, compute_solve_time
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
#Angles per quadrant cases.
angles_per_quadrant = [1,2,3,4,5,6]
num_angular_test = len(angles_per_quadrant)
spatial_discretization = [2,3,4,5,6,7,8,9,10]
num_spatial_test = len(spatial_discretization)

#The two by two case.
computation_time = np.empty([num_angular_test,num_spatial_test])

for angle in range(0,num_angular_test):
  num_angles = int(angles_per_quadrant[angle])
  for space in range(0,num_spatial_test):
    
    num_space = int(spatial_discretization[space])
    numrow = num_space
    numcol = num_space
    num_subsets = numrow*numcol
    numrow = num_space
    numcol = num_space
    num_subsets = numrow*numcol
    print("Angle: "+str(num_angles), "Space: "+str(num_space))
    
    x_cuts = list(np.genfromtxt("x_cuts_"+str(numcol)+".csv",delimiter=","))
    y_cuts = np.genfromtxt("y_cuts_"+str(numrow)+".csv",delimiter=",")
    
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
    time_to_soln = compute_solve_time(graphs)[1]
    print(time_to_soln)
    computation_time[angle][space] = time_to_soln

#computation_time_2 = np.empty([num_angular_test])
#print("2x2 Case")
#for angle in range(0,num_angular_test):
#  num_angles = int(angles_per_quadrant[angle])
#  num_space = 2
#  numrow = num_space
#  numcol = num_space
#  num_subsets = numrow*numcol
#  
#  #The subset_boundaries.
#  subset_boundaries = build_global_subset_boundaries(numcol-1,numrow-1,x_cuts,y_cuts)
#  #Building the adjacency matrix.
#  adjacency_matrix = build_adjacency(subset_boundaries,numcol-1,numrow-1,y_cuts)
#  #Building the graphs
#  graphs = build_graphs(adjacency_matrix,numrow,numcol,num_angles)
#  num_graphs = len(graphs)
#  #Dummy values for the purpose of this test case. 
#  cells_per_subset = [1 for n in range(0,num_subsets)]
#  bdy_cells_per_subset = [[1,1] for n in range(0,num_subsets)]
#  
#  #Adding the universal cost.
#  graphs,time_to_solve = add_edge_cost(graphs,subset_boundaries,cells_per_subset,bdy_cells_per_subset,machine_params,numrow,numcol,True)
#  
#  graphs = pipeline_offset(graphs,num_angles,time_to_solve)
#  
#  graphs = make_edges_universal(graphs)
#  
#  graphs = add_conflict_weights(graphs,time_to_solve,num_angles,True)
#  time_to_soln = compute_solve_time(graphs)[1]
#  print(time_to_soln)
#  computation_time_2[angle] = time_to_soln
#
##3x3 Case
#print("3x3 Case")
#x_cuts = list(np.genfromtxt("x_cuts_3.csv",delimiter=","))
#y_cuts = np.genfromtxt("y_cuts_3.csv",delimiter=",")
#for angle in range(0,num_angular_test):
#  num_angles = int(angles_per_quadrant[angle])
#  num_space = 3
#  numrow = num_space
#  numcol = num_space
#  num_subsets = numrow*numcol
#  
#  #The subset_boundaries.
#  subset_boundaries = build_global_subset_boundaries(numcol-1,numrow-1,x_cuts,y_cuts)
#  #Building the adjacency matrix.
#  adjacency_matrix = build_adjacency(subset_boundaries,numcol-1,numrow-1,y_cuts)
#  #Building the graphs
#  graphs = build_graphs(adjacency_matrix,numrow,numcol,num_angles)
#  num_graphs = len(graphs)
#  #Dummy values for the purpose of this test case. 
#  cells_per_subset = [1 for n in range(0,num_subsets)]
#  bdy_cells_per_subset = [[1,1] for n in range(0,num_subsets)]
#  
#  #Adding the universal cost.
#  graphs,time_to_solve = add_edge_cost(graphs,subset_boundaries,cells_per_subset,bdy_cells_per_subset,machine_params,numrow,numcol,True)
#  
#  graphs = pipeline_offset(graphs,num_angles,time_to_solve)
#  
#  graphs = make_edges_universal(graphs)
#  
#  graphs = add_conflict_weights(graphs,time_to_solve,num_angles,True)
#  time_to_soln = compute_solve_time(graphs)[1]
#  print(time_to_soln)
#  computation_time_2[angle] = time_to_soln
#  
#print("4x4 Case")
#x_cuts = list(np.genfromtxt("x_cuts_4.csv",delimiter=","))
#y_cuts = np.genfromtxt("y_cuts_4.csv",delimiter=",")
#for angle in range(0,num_angular_test):
#  num_angles = int(angles_per_quadrant[angle])
#  num_space = 4
#  numrow = num_space
#  numcol = num_space
#  num_subsets = numrow*numcol
#  
#  #The subset_boundaries.
#  subset_boundaries = build_global_subset_boundaries(numcol-1,numrow-1,x_cuts,y_cuts)
#  #Building the adjacency matrix.
#  adjacency_matrix = build_adjacency(subset_boundaries,numcol-1,numrow-1,y_cuts)
#  #Building the graphs
#  graphs = build_graphs(adjacency_matrix,numrow,numcol,num_angles)
#  num_graphs = len(graphs)
#  #Dummy values for the purpose of this test case. 
#  cells_per_subset = [1 for n in range(0,num_subsets)]
#  bdy_cells_per_subset = [[1,1] for n in range(0,num_subsets)]
#  
#  #Adding the universal cost.
#  graphs,time_to_solve = add_edge_cost(graphs,subset_boundaries,cells_per_subset,bdy_cells_per_subset,machine_params,numrow,numcol,True)
#  
#  graphs = pipeline_offset(graphs,num_angles,time_to_solve)
#  
#  graphs = make_edges_universal(graphs)
#  
#  graphs = add_conflict_weights(graphs,time_to_solve,num_angles,True)
#  time_to_soln = compute_solve_time(graphs)[1]
#  print(time_to_soln)
#  computation_time_2[angle] = time_to_soln