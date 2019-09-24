import numpy as np
import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
from sweep_solver import optimized_tts_numerical,unpack_parameters
from mesh_processor import create_2d_cuts
from optimizer import create_parameter_space,create_bounds,create_constraints,get_column_cdf,create_opt_cut_suite,get_highest_jumps
from scipy.optimize import basinhopping, minimize
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
plt.close("all")
#Communication time per double
t_comm = 4.47e-09
#The number of bytes to communicate per subset.
#The message latency time.
m_l = 1
latency = 4110.0e-09
#Solve time per cell..
Tc = 1208.383e-09
upc = 4.0
upbc = 2.0
Twu = 147.0754e-09
Tm = 65.54614e-09
Tg = 175.0272e-09
mcff = 1.181
machine_parameters = (Twu,Tc,Tm,Tg,upc,upbc,mcff,t_comm,latency,m_l)
num_angles = 1
Am = 36
unweighted = True
Ay = 1

points = np.genfromtxt("unbalanced_pins_sparse_centroid_data").T

gxmin = 0.0
gxmax = 10.0
gymin = 0.0
gymax = 10.0

numrow = 10
numcol = 10

verts = np.genfromtxt("unbalanced_pins_sparse_vert_data")
x_values_func = get_highest_jumps(verts[:,0],gxmin,gxmax,numcol)
#x_values,y_cut_suite = create_opt_cut_suite(verts,gxmin,gxmax,gymin,gymax,numcol,numrow)