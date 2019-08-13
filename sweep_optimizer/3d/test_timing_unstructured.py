import numpy as np
from mesh_processor import create_2d_cuts,create_3d_cuts,get_z_cells
from optimizer import create_parameter_space,create_parameter_space_3d
from sweep_solver import optimized_tts,optimized_tts_3d,optimized_tts_numerical
from scipy.optimize import minimize,basinhopping
import time


f = lambda x,y,z: 1
#The machine parameters.
#Communication time per double
t_comm = 4.47e-09
#The number of bytes to communicate per subset.
#The message latency time.
m_l = 1
latency = 4110.0e-09
#Solve time per cell.
Tc = 179.515e-09
upc = 4.0
upbc = 4.0
Twu = 518.19882e-09
Tm = 53.1834e-09
Tg = 12.122e-09
mcff = 132.0

machine_parameters = (Twu,Tc,Tm,Tg,upc,upbc,mcff,t_comm,latency,m_l)


#Number of rows and columns and planes.
num_angles = 1
Am = 1
unweighted=True

num_row = 5
num_col = 5

xmin = 0.0
xmax = 1.0
ymin = 0.0
ymax = 1.0

x_cuts,y_cuts = create_2d_cuts(xmin,xmax,num_col,ymin,ymax,num_row)
points = np.genfromtxt("unbalanced_pins_centroid_data").T
params = create_parameter_space(x_cuts,y_cuts,num_row,num_col)

max_time_reg = optimized_tts_numerical(params, points,xmin,xmax,ymin,ymax,num_row,num_col,machine_parameters,num_angles,Am,unweighted)