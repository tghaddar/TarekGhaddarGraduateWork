import numpy as np
import sys
#sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
sys.path.append(r'C:\Users\tghad\Documents\GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
from sweep_solver import optimized_tts_numerical,unpack_parameters,plot_subset_boundaries_2d
from mesh_processor import create_2d_cuts,get_cells_per_subset_2d_test,get_cells_per_subset_2d_robust,get_cells_per_subset_2d_more_robust
from optimizer import create_parameter_space,create_bounds,create_constraints,get_column_cdf,create_opt_cut_suite,get_highest_jumps
from scipy.optimize import basinhopping, minimize
import matplotlib.pyplot as plt
from build_global_subset_boundaries import build_global_subset_boundaries
from build_adjacency_matrix import build_adjacency
from shapely.geometry import Point,Polygon
plt.close("all")

gxmin = 0.0
gxmax = 152.0
gymin = 0.0
gymax = 54.994
numrow = 13
numcol = 42
add_cells = True
points = np.genfromtxt("level2centroids").T
verts = np.genfromtxt("level2_vert_data")
f = open("lvl2_cell_verts",'r')
level2_cell_data = [line.split() for line in f]
for i in range(0,len(level2_cell_data)):
  level2_cell_data[i] = [int(x) for x in level2_cell_data[i]]


x_cuts,y_cuts = create_2d_cuts(gxmin,gxmax,numcol,gymin,gymax,numrow)
x_cuts = np.genfromtxt("lvl2_42_reg_x_cuts")
y_cuts = np.genfromtxt("lvl2_13_reg_y_cuts")
boundaries = build_global_subset_boundaries(numcol-1,numrow-1,x_cuts,y_cuts)
adjacency_matrix = build_adjacency(boundaries,numcol-1,numrow-1,y_cuts)
level2_reg_cut_cell_data = np.genfromtxt("level2_reg_cut_cell_data.txt")
cells_per_subset,bcps = get_cells_per_subset_2d_test(points,boundaries,adjacency_matrix,numrow,numcol,False)
#cells_per_subset,bcps = get_cells_per_subset_2d_robust(points,level2_cell_data,verts,boundaries,adjacency_matrix,numrow,numcol,True)

cells_per_subset,bcps = get_cells_per_subset_2d_more_robust(points,level2_cell_data,verts,boundaries,adjacency_matrix,numrow,numcol,True)




num_cell_pdt = sum(level2_reg_cut_cell_data)
num_cell = sum(cells_per_subset)

percent_diff = []
subsets = []
for i in range(0,len(cells_per_subset)):
  subsets.append(i)
  percent_diff.append(abs(cells_per_subset[i] - level2_reg_cut_cell_data[i])/level2_reg_cut_cell_data[i]*100)
  
pdiff = np.array(percent_diff)
#Get cells in the subset. Grab centroids in the cell. grab vert data for the cell. If cut line coordinate is in the cell, then you add that to the number of boundary cells. Add a cell to the appropriate neighboring subset.
