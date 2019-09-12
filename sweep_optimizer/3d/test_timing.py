import numpy as np
from mesh_processor import create_2d_cuts,create_3d_cuts,get_z_cells
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
#Solve time per cell.
Tc = 2683.769129e-09
upc = 8.0
upbc = 4.0
Twu = 5779.92891144936e-09
Tm = 111.971932555903e-09
Tg = 559.127e-09
mcff = 1.32

machine_parameters = (Twu,Tc,Tm,Tg,upc,upbc,mcff,t_comm,latency,m_l)


#Number of rows and columns and planes.
numcol = 2
numrow = 2
numplane = 2
num_angles = 1
Am = 10
unweighted=True


#Global boundaries.
global_xmin = 0.0
global_xmax = 8.0
global_ymin = 0.0
global_ymax = 8.0
global_zmin = 0.0
global_zmax = 8.0

#An adjusted Az for regular cases that normalizes the boundary cost for each processor so it matches the performance model.
Az = global_zmax/2
mult = Az/1.939
#Az = 1

z_cuts,x_cuts,y_cuts = create_3d_cuts(global_xmin,global_xmax,numcol,global_ymin,global_ymax,numrow,global_zmin,global_zmax,numplane)
params = create_parameter_space_3d(x_cuts,y_cuts,z_cuts,numrow,numcol,numplane)
num_params = len(params)

start = time.time()
max_time = optimized_tts_3d(params,f,global_xmin,global_xmax,global_ymin,global_ymax,global_zmin,global_zmax,numrow,numcol,numplane,machine_parameters,num_angles,Am,Az,unweighted,True)
end = time.time()
print(end-start)
