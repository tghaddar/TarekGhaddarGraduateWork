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

gxmin = 0.0
gxmax = 152.0
gymin = 0.0
gymax = 54.994

#Communication time per double
t_comm = 4.47e-09
#The number of bytes to communicate per subset.
#The message latency time.
m_l = 90
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
add_cells=True
Ay = 1
numcol = 42
numrow = 13
verts = np.genfromtxt("level2_vert_data")
points = np.genfromtxt("level2centroids").T
f = open("lvl2_cell_verts",'r')
level2_cell_data = [line.split() for line in f]
for i in range(0,len(level2_cell_data)):
  level2_cell_data[i] = [int(x) for x in level2_cell_data[i]]
#Trying optimizing the spiderweb.

pdt_data = np.genfromtxt("level2_opt_sweep_data.txt")
pdt_data = np.reshape(pdt_data,(5,10))
pdt_data_median = np.empty(5)
pdt_data_min = np.empty(5)
pdt_data_max = np.empty(5)
for i in range(0,5):
  median = np.median(pdt_data[i])
  pdt_data_median[i] = median
  pdt_data_min[i] = np.min(pdt_data[i])
  pdt_data_max[i] = np.max(pdt_data[i])

max_times = []

x_cuts_reg = np.genfromtxt("level2_reg_cuts/lvl2_42_reg_x_cuts")
y_cuts_reg = np.genfromtxt("level2_reg_cuts/lvl2_13_reg_y_cuts")
params = create_parameter_space(x_cuts_reg,y_cuts_reg,numrow,numcol)
max_times.append(optimized_tts_numerical(params,level2_cell_data,verts,gxmin,gxmax,gymin,gymax,numrow,numcol,machine_parameters,num_angles,Am,Ay,add_cells,unweighted))



x_cuts_balanced = [0.0,7.0,14.62,16.1565,17.16,18.1635,19.7,30.5,38.76,47.9,55.52,64.66,67.835,68.47,69.105,69.74,71.53,71.78,72.03,72.28,73.27,74.26,74.92,75.58,76.24,76.9,77.89,78.88,79.13,79.38,79.63,81.42,82.055,82.69,83.325,86.5,95.64,103.26,112.4,120.66,130.88,141.44,gxmax]
y_cuts_lbd_col = [0.0,19.1775,31.228,43.8345,47.0373,48.0957,48.7307,49.7507,51.194,51.5273,52.024,53.014,54.04,54.994]
y_cuts_balanced = []
for col in range(0,numcol):
  y_cuts_balanced.append(y_cuts_lbd_col)
params = create_parameter_space(x_cuts_balanced,y_cuts_balanced,numrow,numcol)
max_times.append(optimized_tts_numerical(params,level2_cell_data,verts,gxmin,gxmax,gymin,gymax,numrow,numcol,machine_parameters,num_angles,Am,Ay,add_cells,unweighted))


x_cuts_lb = np.genfromtxt("level2_lb_cuts/cut_line_data_x_42")
y_cuts_lb = np.genfromtxt("level2_lb_cuts/cut_line_data_y_42")
params = create_parameter_space(x_cuts_lb,y_cuts_lb,numrow,numcol)
max_times.append(optimized_tts_numerical(params,level2_cell_data,verts,gxmin,gxmax,gymin,gymax,numrow,numcol,machine_parameters,num_angles,Am,Ay,add_cells,unweighted))



x_cuts_lbd = np.genfromtxt("level2_lbd_cuts/cut_line_data_x_42")
y_cuts_lbd = np.genfromtxt("level2_lb_cuts/cut_line_data_y_42")
params = create_parameter_space(x_cuts_lbd,y_cuts_lbd,numrow,numcol)
max_times.append(optimized_tts_numerical(params,level2_cell_data,verts,gxmin,gxmax,gymin,gymax,numrow,numcol,machine_parameters,num_angles,Am,Ay,add_cells,unweighted))


x_cuts_opt = np.genfromtxt("level2_opt_x_cuts")
y_cuts_opt = np.genfromtxt("level2_opt_y_cuts")
params = create_parameter_space(x_cuts_opt,y_cuts_opt,numrow,numcol)
max_times.append(optimized_tts_numerical(params,level2_cell_data,verts,gxmin,gxmax,gymin,gymax,numrow,numcol,machine_parameters,num_angles,Am,Ay,add_cells,unweighted))

np.savetxt("tts_level2_data",max_times)

