import numpy as np
import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
#sys.path.append(r'C:\Users\tghad\Documents\GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
from sweep_solver import optimized_tts_numerical,unpack_parameters,plot_subset_boundaries_2d
from mesh_processor import create_2d_cuts,get_cells_per_subset_2d_test,get_cells_per_subset_2d_robust
from optimizer import create_parameter_space,create_bounds,create_constraints,get_column_cdf,create_opt_cut_suite,get_highest_jumps
from scipy.optimize import basinhopping, minimize
import matplotlib.pyplot as plt
from build_global_subset_boundaries import build_global_subset_boundaries
from build_adjacency_matrix import build_adjacency
from shapely.geometry import MultiPoint
import time
plt.close("all")

gxmin = 0.0
gymin = 0.0
gxmax = 10.0
gymax = 10.0
numrow = 7
numcol = 7
subset = [i for i in range(0,numrow*numcol)]
add_cells = True
#points = np.genfromtxt("level2centroids").T

pin_verts = np.genfromtxt("unbalanced_pins_sparse_vert_data")  
f = open("sparse_pins_cell_verts",'r')
sparse_pins_cell_data = [line.split() for line in f]
for i in range(0,len(sparse_pins_cell_data)):
  sparse_pins_cell_data[i] = [int(x) for x in sparse_pins_cell_data[i]]


#x_cuts,y_cuts = create_2d_cuts(gxmin,gxmax,numcol,gymin,gymax,numrow)
x_cuts = np.genfromtxt("spiderweb_lbd_cut_line_data/cut_line_data_x_7")
y_cuts = np.genfromtxt("spiderweb_lbd_cut_line_data/cut_line_data_y_7")
boundaries = build_global_subset_boundaries(numcol-1,numrow-1,x_cuts,y_cuts)
adjacency_matrix = build_adjacency(boundaries,numcol-1,numrow-1,y_cuts)
cells_per_subset,bcps = get_cells_per_subset_2d_robust(sparse_pins_cell_data,pin_verts,boundaries,adjacency_matrix,numrow,numcol)

pdt_cell_data = np.genfromtxt("spiderweb_7x7_lbd_cell_output.txt")

plt.figure()
plt.xlabel("Subset ID")
plt.ylabel("Cell Count")
plt.plot(subset,pdt_cell_data,'o',label="PDT")
plt.plot(subset,cells_per_subset,'x',label="TTS")
plt.legend(loc='best')
plt.savefig("../../figures/spiderweb_cell_comp_7x7.pdf")
