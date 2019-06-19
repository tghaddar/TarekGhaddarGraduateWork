import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
import numpy as np
from sweep_solver import optimized_tts_numerical,unpack_parameters,plot_subset_boundaries_2d
from mesh_processor import create_2d_cuts
from build_global_subset_boundaries import build_global_subset_boundaries
from optimizer import create_parameter_space,create_bounds
from scipy.optimize import minimize,basinhopping
import time
num_row = 4
num_col = 4
xmin = 0.0
xmax = 1.0
ymin = 0.0
ymax = 1.0

x_cuts,y_cuts = create_2d_cuts(xmin,xmax,num_col,ymin,ymax,num_row)
boundaries = build_global_subset_boundaries(num_row-1,num_col-1,x_cuts,y_cuts)
plot_subset_boundaries_2d(boundaries,num_row*num_col,"../../figures/theoretical_plot.pdf")
