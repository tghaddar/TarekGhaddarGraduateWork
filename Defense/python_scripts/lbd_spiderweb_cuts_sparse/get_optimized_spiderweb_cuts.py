
import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
import numpy as np
from sweep_solver import optimized_tts_numerical,unpack_parameters
from mesh_processor import create_2d_cuts
from optimizer import create_parameter_space,create_bounds
from scipy.optimize import minimize,basinhopping
import time

#Communication time per double
t_comm = 4.47e-09
#The number of bytes to communicate per subset.
#The message latency time.
m_l = 1
latency = 4110.0e-09
#Solve time per unknown.
t_u = 3500.0e-09
upc = 4.0
upbc = 2.0
machine_params = (t_u,upc,upbc,t_comm,latency,m_l)

ns = [6]
num_suite = len(ns)

num_angles = 1
unweighted = True
xmin = 0.0
xmax = 10.0
ymin = 0.0
ymax = 10.0
points = np.genfromtxt("../unbalanced_pins_sparse_centroid_data").T
mts = [None]*num_suite
opt_mts = [None]*num_suite

for i in range(0,num_suite):
  s = ns[i]
  print(s)
  num_row = s
  num_col = s
  
#  x_cuts,y_cuts = create_2d_cuts(xmin,xmax,num_col,ymin,ymax,num_row)
  x_cuts = np.genfromtxt("cut_line_data_x_"+str(s))
  y_cuts = np.genfromtxt("cut_line_data_y_"+str(s))
  params = create_parameter_space(x_cuts,y_cuts,num_row,num_col)
  num_params = len(params)
  mt = optimized_tts_numerical(params,points,xmin,xmax,ymin,ymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)
  mts[i] = mt
  
  bounds = create_bounds(num_params,xmin,xmax,ymin,ymax,num_row,num_col)
  args = (points,xmin,xmax,ymin,ymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)
  start = time.time()
  max_time = basinhopping(optimized_tts_numerical,params,niter=10,T=1.0,stepsize=0.1,minimizer_kwargs={"args":args,"bounds":bounds,"method":"SLSQP","options":{'eps':0.01}})
  end = time.time()
  print(end-start)
  opt_mts[i] = max_time.fun

