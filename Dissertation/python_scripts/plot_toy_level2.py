import numpy as np
import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
from sweep_solver import optimized_tts_numerical,unpack_parameters,plot_subset_boundaries_2d
from mesh_processor import create_2d_cuts
from optimizer import create_parameter_space,create_bounds,create_constraints,get_column_cdf,create_opt_cut_suite,get_opt_cut_suite_best,get_highest_jumps
from scipy.optimize import basinhopping, minimize
import matplotlib.pyplot as plt
from build_global_subset_boundaries import build_global_subset_boundaries
import itertools
plt.close("all")

gxmin = 0.0
gxmax = 152.0
gymin = 0.0
gymax = 54.994
numrow = 13
numcol = 42

x_cuts_lb = [0.0,7.0,14.62,16.1565,17.16,18.1635,19.7,30.5,38.76,47.9,55.52,64.66,67.835,68.47,69.105,69.74,71.53,71.78,72.03,72.28,73.27,74.26,74.92,75.58,76.24,76.9,77.89,78.88,79.13,79.38,79.63,81.42,82.055,82.69,83.325,86.5,95.64,103.26,112.4,120.66,130.88,141.44,gxmax]
y_cuts_lbd_col = [0.0,19.1775,31.228,43.8345,47.0373,48.0957,48.7307,49.7507,51.194,51.5273,52.024,53.014,54.04,54.994]
y_cuts_lb = []
for col in range(0,numcol):
  y_cuts_lb.append(y_cuts_lbd_col)
boundaries = build_global_subset_boundaries(numcol-1,numrow-1,x_cuts_lb,y_cuts_lb)

plot_subset_boundaries_2d(boundaries,numcol*numrow,[],'hand_balance')


x_cuts_bin = np.genfromtxt("level2_best_x_cuts")
y_cuts_bin = np.genfromtxt("level2_best_y_cuts")

boundaries = build_global_subset_boundaries(numcol-1,numrow-1,x_cuts_bin,y_cuts_bin)

plot_subset_boundaries_2d(boundaries,numcol*numrow,[],'bin_tree')