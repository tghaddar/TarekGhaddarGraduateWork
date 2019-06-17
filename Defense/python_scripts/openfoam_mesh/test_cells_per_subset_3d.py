#Test cells/subset
import numpy as np
import sys
sys.path.append(r'C:\Users\tghad\Documents\GitHub\TarekGhaddarGraduateWork\sweep_optimizer\3d')
from mesh_processor import get_cells_per_subset_3d_numerical,create_3d_cuts,get_cells_per_subset_3d_numerical_test
import build_3d_adjacency as b3a
import time


xmin = 0.0
xmax = 60.96
ymin = 0.0
ymax = xmax
zmin = xmin
zmax = 146.05


z_cuts,x_cuts,y_cuts = create_3d_cuts(xmin,xmax,5,ymin,ymax,5,zmin,zmax,5)
boundaries = b3a.build_3d_global_subset_boundaries(4,4,4,x_cuts,y_cuts,z_cuts)

points = np.genfromtxt("im1_cell_centers").T

start = time.time()
cells_per_subset,bdy_cells_per_subset = get_cells_per_subset_3d_numerical(points,boundaries)
end = time.time()
print(end-start)

start = time.time()
cells_per_subset_test,bdy_cells_per_subset_test = get_cells_per_subset_3d_numerical_test(points,boundaries)
end = time.time()
print(end-start)