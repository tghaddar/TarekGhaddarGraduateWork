import numpy as np
import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
from sweep_solver import optimized_tts_numerical,unpack_parameters
from mesh_processor import create_2d_cuts
from optimizer import create_parameter_space,create_bounds,create_constraints,get_column_cdf,create_opt_cut_suite,get_opt_cut_suite_best,get_highest_jumps
from scipy.optimize import basinhopping, minimize
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
plt.close("all")
#Communication time per double
t_comm = 4.47e-09
#The number of bytes to communicate per subset.
#The message latency time.
m_l = 50
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

verts = np.genfromtxt("more_sparse_data/vert_data")
f = open("lvl2_cell_verts",'r')
level2_cell_data = [line.split() for line in f]
for i in range(0,len(level2_cell_data)):
  level2_cell_data[i] = [int(x) for x in level2_cell_data[i]]
  
f = open("more_sparse_data/cell_verts",'r')
sparse_pins_cell_data = [line.split() for line in f]
for i in range(0,len(sparse_pins_cell_data)):
  sparse_pins_cell_data[i] = [int(x) for x in sparse_pins_cell_data[i]]

gxmin = 0.0
gxmax = 10.0
gymin = 0.0
gymax = 10.0

numrows = [2,3,4,5,6,7,8,9,10]
numcols = [2,3,4,5,6,7,8,9,10]
  
x_values = get_highest_jumps(verts[:,0],gxmin,gxmax,10)

max_times_case = {}
max_times_case_time_only = []

fs_case_only = []


for i in range(0,len(numrows)):
  numrow = numrows[i]
  numcol = numcols[i]
  
  x_values,y_cut_suite = create_opt_cut_suite(verts,gxmin,gxmax,gymin,gymax,numcol,numrow)
  
  max_times = []
  fs = []
  add_cells = False
  for j in range(0,len(y_cut_suite)):
    x_cuts = x_values
    y_cuts = y_cut_suite[j]
    params = create_parameter_space(x_cuts,y_cuts,numrow,numcol)
    max_time,f = optimized_tts_numerical(params,sparse_pins_cell_data,verts,gxmin,gxmax,gymin,gymax,numrow,numcol,machine_parameters,num_angles,Am,Ay,add_cells,unweighted)
    max_times.append(max_time)
    fs.append(f)
  
  min_index = max_times.index(min(max_times))
  y_cuts_min = y_cut_suite[min_index]
  x_cuts_min = x_values
  
  max_times_case[numrow] = (min(max_times),x_cuts_min,y_cuts_min)
  max_times_case_time_only.append(min(max_times))
  fs_case_only.append(fs[min_index])