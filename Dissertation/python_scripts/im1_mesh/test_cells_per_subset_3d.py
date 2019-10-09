#Test cells/subset
import numpy as np
import sys
sys.path.append(r'C:\Users\tghad\Documents\GitHub\TarekGhaddarGraduateWork\sweep_optimizer\3d')
from mesh_processor import get_cells_per_subset_3d_numerical,create_3d_cuts,get_cells_per_subset_3d_numerical_test2,create_2d_cuts
from mesh_processor import get_cells_per_subset_2d_test, get_cells_per_subset_2d_numerical
import build_3d_adjacency as b3a
from build_global_subset_boundaries import build_global_subset_boundaries
from build_adjacency_matrix import build_adjacency
import time


#xmin = 0.0
#xmax = 60.96
#ymin = 0.0
#ymax = xmax
#zmin = xmin
#zmax = 146.05
xmin = 0.0
xmax = 1.0
ymin = 0.0
ymax = 1.0


z_cuts,x_cuts,y_cuts = create_3d_cuts(xmin,xmax,5,ymin,ymax,5,zmin,zmax,5)
boundaries = b3a.build_3d_global_subset_boundaries(4,4,4,x_cuts,y_cuts,z_cuts)

x_cuts,y_cuts = create_2d_cuts(xmin,xmax,2,ymin,ymax,2)
boundaries = build_global_subset_boundaries(1,1,x_cuts,y_cuts)
adjacency_matrix = build_adjacency(boundaries,1,1,y_cuts)

points = np.genfromtxt("../unbalanced_pins_centroid_data").T

#start = time.time()
#cells_per_subset,bdy_cells_per_subset = get_cells_per_subset_3d_numerical(points,boundaries)
#end = time.time()
#print(end-start)

#start = time.time()
#cells_per_subset_test,bdy_cells_per_subset_test = get_cells_per_subset_3d_numerical_test2(points,boundaries)
#end = time.time()
#print(end-start)

start = time.time()
cells_per_subset,bdy_cells_per_sub = get_cells_per_subset_2d_test(points,boundaries,adjacency_matrix,2,2)
end = time.time()

print(end - start)
start = time.time()
cells_per_subset_old,bdy_cells_per_sub_old = get_cells_per_subset_2d_numerical(points,boundaries,adjacency_matrix,2,2)
end = time.time()
print(end-start)