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
numcols = [1,2,8,16,32,32,64]
numrows =  [1,2,4,16,16,32,32]
numcols = [1,2,8,16]
numrows =  [1,2,4,16]
numplane = 1
num_angles = 1
Am = 10
unweighted=True
test = False


#Global boundaries.
global_xmin = 0.0
global_xmaxs = [16.0,32.0,64.0,128.0,128.0,256.0,256.0]
global_xmaxs = [16.0,32.0,64.0,128.0]
global_ymin = 0.0
global_ymaxs = [16.0,32.0,64.0,128.0,128.0,128.0,256.0]
global_ymaxs = [16.0,32.0,64.0,128.0]
global_zmin = 0.0
global_zmaxs = [16.0,32.0,64.0,128.0,256.0,256.0,256.0]
global_zmaxs = [16.0,32.0,64.0,128.0]


max_times = []

for i in range(0,len(numcols)):
  
  global_xmax = global_xmaxs[i]
  global_ymax = global_ymaxs[i]
  global_zmax = global_zmaxs[i]
  
  numcol = 0
  numrow = 0
  #An adjusted Az for regular cases that normalizes the boundary cost for each processor so it matches the performance model.
  Az = global_zmax/2
  if i == 0:
    numcol = 1
    numrow = 1
    numplane = 1
    Az = 1
    mcff = 1
  else:
    numcol = numcols[i]
    numrow = numrows[i]
    numplane = 2
    mcff = 1.32

  machine_parameters = (Twu,Tc,Tm,Tg,upc,upbc,mcff,t_comm,latency,m_l)
  
  z_cuts,x_cuts,y_cuts = create_3d_cuts(global_xmin,global_xmax,numcol,global_ymin,global_ymax,numrow,global_zmin,global_zmax,numplane)
  params = create_parameter_space_3d(x_cuts,y_cuts,z_cuts,numrow,numcol,numplane)
  num_params = len(params)
  
  start = time.time()
  max_time = optimized_tts_3d(params,f,global_xmin,global_xmax,global_ymin,global_ymax,global_zmin,global_zmax,numrow,numcol,numplane,machine_parameters,num_angles,Am,Az,unweighted,test)
  end = time.time()
  print(end-start)
  max_times.append(max_time)


timing_csv = np.savetxt("3d_timing_runs.csv",max_times,delimiter=',')