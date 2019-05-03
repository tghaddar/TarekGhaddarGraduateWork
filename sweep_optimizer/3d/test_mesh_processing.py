from mesh_processor import analytical_mesh_integration_2d,create_2d_cuts,get_cells_per_subset_2d
from build_global_subset_boundaries import build_global_subset_boundaries

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

#The subset_boundaries.
subset_boundaries = build_global_subset_boundaries(numcol-1,numrow-1,x_cuts,y_cuts)

f = lambda x,y: 5

cells_per_subset, bdy_cells_per_subset = get_cells_per_subset_2d(f,subset_boundaries)



