import numpy as np
import sys
sys.path.append('/Users/79g/TarekGhaddarGraduateWork/sweep_optimizer/3d')
#sys.path.append(r'C:\Users\tghad\Documents\GitHub\TarekGhaddarGraduateWork\sweep_optimizer\3d')
from sweep_solver import optimized_tts_numerical,unpack_parameters,plot_subset_boundaries_2d
from mesh_processor import create_2d_cuts
from optimizer import create_parameter_space,create_bounds,create_constraints,get_column_cdf,create_opt_cut_suite,get_highest_jumps
from scipy.optimize import basinhopping, minimize
import matplotlib.pyplot as plt
from build_global_subset_boundaries import build_global_subset_boundaries
import itertools
plt.close("all")

gxmin = 0.0
gxmax = 152.0
gymin = 0.0
gymax = 54.994

#Communication time per double
t_comm = 4.47e-09
#The number of bytes to communicate per subset.
#The message latency time.
m_l = 80
latency = 4110.0e-09
#Solve time per cell..
Tc = 1279.7325e-09
upc = 4.0
upbc = 2.0
#Twu = 147.0754e-09
Twu = 197.898e-09
#Tm = 65.54614e-09
Tm = 70.9439e-09
#Tg = 175.0272e-09
Tg = 173.193e-09
mcff = 1.181
machine_parameters = (Twu,Tc,Tm,Tg,upc,upbc,mcff,t_comm,latency,m_l)
#num_angles_vec = [1,2,3,5,10]
num_angles_vec = [1,2,3]
#Am_vec = [90,45,30,18,9]
Am_vec = [90,45,30]
unweighted = True
Ay = 1
numcol = 42
numrow = 13
add_cells = True
verts = np.genfromtxt("level2_vert_data")
f = open("lvl2_cell_verts",'r')
level2_cell_data = [line.split() for line in f]
for i in range(0,len(level2_cell_data)):
  level2_cell_data[i] = [int(x) for x in level2_cell_data[i]]
  
x_cuts = np.genfromtxt("lvl2_42_reg_x_cuts")
y_cuts = np.genfromtxt("lvl2_13_reg_y_cuts")

params = create_parameter_space(x_cuts,y_cuts,numrow,numcol)

max_times_reg = [None]*len(Am_vec)
fs_reg = [None]*len(Am_vec)

for a in range(0, len(Am_vec)):
    Am = Am_vec[a]
    num_angles = num_angles_vec[a]
    max_times_reg[a],fs_reg[a] = optimized_tts_numerical(params,level2_cell_data,verts,gxmin,gxmax,gymin,gymax,numrow,numcol,machine_parameters,num_angles,Am,Ay,add_cells,unweighted)
  
