import numpy as np
import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
from sweep_solver import optimized_tts_numerical,unpack_parameters
from mesh_processor import create_2d_cuts
from optimizer import create_parameter_space,create_bounds
from scipy.optimize import minimize,differential_evolution


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


light_case = 100
mid_case = 1000
heavy_case = 1000

gxmin = 0.0
gxmax = 1.0
gymin = 0.0
gymax = 1.0
num_row = 10
num_col = 10

cols = [0.0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for i in range(0,10):
  xmin = cols[i]
  xmax = cols[i+1]
  ymin = 0.0
  ymax = 0.0
  if i%2 == 0:
    ymin = 0.0
    ymax = 0.25
  else:
    ymin = 0.75
    ymax = gymax
  
  #Setting up the local for this column.
  pointsx = np.random.uniform(xmin,xmax,light_case)
  pointsy = np.random.uniform(ymin,ymax,light_case)
  points_local = np.stack((pointsx,pointsy),axis=0)
  if i == 0:
    points = points_local
  else:
    points = np.append(points,points_local,axis=1)

eps = ((gxmax-gxmin)/num_row)

x_cuts = np.genfromtxt("x_cuts_10_worst.csv",delimiter=",")
y_cuts = np.genfromtxt("y_cuts_10_worst.csv",delimiter=",")
params = create_parameter_space(x_cuts,y_cuts,num_row,num_col)
num_params = len(params)
bounds = create_bounds(num_params,gxmin,gxmax,gymin,gymax,num_row,num_col)
args = (points,gxmin,gxmax,gymin,gymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)
#max_time = minimize(optimized_tts_numerical,params,method='SLSQP',args=args,bounds=bounds,options={'maxiter':1000,'maxfun':1000,'disp':True,'eps':eps},tol=1e-08)
max_time = differential_evolution(optimized_tts_numerical,bounds,args=args)
x_cuts,y_cuts = unpack_parameters(max_time.x,gxmin,gxmax,gymin,gymax,num_col,num_row)
    
  
  
