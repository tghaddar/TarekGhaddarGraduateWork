import numpy as np
from mesh_processor import create_2d_cuts,create_3d_cuts
from optimizer import create_parameter_space,create_parameter_space_3d
from sweep_solver import optimized_tts,optimized_tts_3d
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
#Solve time per unknown.
Tc = 1000.0e-09
upc = 8.0
upbc = 4.0
Twu = 35000.0e-09
Tm = 100.0e-09
Tg = 75.0e-09
mcff = 1.330

machine_parameters = (Twu,Tc,Tm,Tg,upc,upbc,mcff,t_comm,latency,m_l)


#Number of rows and columns and planes.
numcol = 64
numrow = 32
numplane = 2
num_angles = 1
Am = 10
unweighted=True

#Global boundaries.
global_xmin = 0.0
global_xmax = 256.0
global_ymin = 0.0
global_ymax = 256.0
global_zmin = 0.0
global_zmax = 256.0

z_cuts,x_cuts,y_cuts = create_3d_cuts(global_xmin,global_xmax,numcol,global_ymin,global_ymax,numrow,global_zmin,global_zmax,numplane)
params = create_parameter_space_3d(x_cuts,y_cuts,z_cuts,numrow,numcol,numplane)
num_params = len(params)

start = time.time()
max_time = optimized_tts_3d(params,f,global_xmin,global_xmax,global_ymin,global_ymax,global_zmin,global_zmax,numrow,numcol,numplane,machine_parameters,num_angles,Am,unweighted,False)
end = time.time()
print(end-start)
