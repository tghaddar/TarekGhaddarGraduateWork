import numpy as np
import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
from mesh_processor import create_3d_cuts
from optimizer import create_parameter_space_3d
from optimizer import create_bounds_3d
from sweep_solver import optimized_tts_3d_numerical
from scipy.optimize import minimize
import time

gxmin = 0.0
gxmax = 60.96
gymin = 0.0
gymax = 60.96
gzmin = 0.0
gzmax = 146.05

#The machine parameters.
#Communication time per double
t_comm = 4.47e-09
#The number of bytes to communicate per subset.
#The message latency time.
m_l = 1
latency = 4110.0e-09
#Solve time per unknown.
t_u = 450.0e-09
upc = 4.0
upbc = 2.0
machine_params = (t_u,upc,upbc,t_comm,latency,m_l)

num_angles = 1
unweighted = True
test = False
num_row = 5
num_col = 5
num_plane = 5

im1points = np.genfromtxt("im1_cell_centers").T
z_cuts,x_cuts,y_cuts = create_3d_cuts(gxmin,gxmax,num_col,gymin,gymax,num_row,gzmin,gzmax,num_plane)
params = create_parameter_space_3d(x_cuts,y_cuts,z_cuts,num_row,num_col,num_plane)
num_params = len(params)
bounds = create_bounds_3d(num_params,gxmin,gxmax,gymin,gymax,gzmin,gzmax,num_row,num_col,num_plane)
args = (im1points,gxmin,gxmax,gymin,gymax,gzmin,gzmax,num_row,num_col,num_plane,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted,test)
eps = 0.01
start = time.time()
max_time = minimize(optimized_tts_3d_numerical,params,method='SLSQP',args=args,bounds=bounds,options={'maxiter':500,'maxfun':500,'disp':True,'eps':eps},tol=1e-07)
end = time.time()