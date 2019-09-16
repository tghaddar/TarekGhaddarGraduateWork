import numpy as np
from mesh_processor import create_2d_cuts,create_3d_cuts,get_z_cells
from optimizer import create_parameter_space,create_parameter_space_3d
from sweep_solver import optimized_tts,optimized_tts_3d,optimized_tts_numerical,optimized_tts
from scipy.optimize import minimize,basinhopping
import time



f = lambda x,y: 1
#The machine parameters.
#Communication time per double
t_comm = 4.47e-09
#The number of bytes to communicate per subset.
#The message latency time.
m_l = 150
latency = 4110.0e-09
#Solve time per cell.
Tc = 179.515e-09
upc = 4.0
upbc = 2.0
Twu = 518.19882e-09
Tm = 53.1834e-09
Tg = 12.122e-09
mcff = 1.036

machine_parameters = (Twu,Tc,Tm,Tg,upc,upbc,mcff,t_comm,latency,m_l)

num_rows = [2,4,8,16,32]
num_cols = [2,4,8,16,32]
num_rows = [4]
num_cols=[4]
unweighted = True
num_angles = 1
Am = 10

xmin = 0.0
xmaxs = [32.0,64.0,128.0,256.0,512.0]
xmaxs = [16.0]
ymin = 0.0
ymaxs = [32.0,64.0,128.0,256.0,512.0]
ymaxs=[16.0]

times = [None]*len(num_rows)

for i in range(0,len(num_rows)):
  num_row = num_rows[i]
  num_col = num_cols[i]
  xmax = xmaxs[i]
  ymax = ymaxs[i]
  Ay = ymax/num_row

  x_cuts,y_cuts = create_2d_cuts(xmin,xmax,num_col,ymin,ymax,num_row)
  params = create_parameter_space(x_cuts,y_cuts,num_row,num_col)
  
  max_time_reg = optimized_tts(params, f,xmin,xmax,ymin,ymax,num_row,num_col,machine_parameters,num_angles,Am,Ay,unweighted)
  times[i] = max_time_reg