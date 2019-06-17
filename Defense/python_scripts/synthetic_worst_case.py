import numpy as np
import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
from sweep_solver import optimized_tts_numerical,unpack_parameters
from mesh_processor import create_2d_cuts
from optimizer import create_parameter_space,create_bounds
from scipy.optimize import minimize,differential_evolution
from build_global_subset_boundaries import build_global_subset_boundaries
import matplotlib.pyplot as plt
from copy import copy
plt.close("all")
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

def jacobian(cuts,f,global_xmin,global_xmax,global_ymin,global_ymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted):
  num_cuts = len(cuts)
  eps = 0.01
  alphas = np.empty(num_cuts)
  
  for cut in range(0,num_cuts):
    test_cuts = copy(cuts)
    test_cuts[cut] += eps
    print("jac cuts: ", test_cuts)
    perturbed_soln = optimized_tts_numerical(test_cuts,f,global_xmin,global_xmax,global_ymin,global_ymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)
    soln = optimized_tts_numerical(cuts,f,global_xmin,global_xmax,global_ymin,global_ymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)
    print(perturbed_soln,soln)
    alphas[cut] = (perturbed_soln - soln)/eps
  
  return alphas


x_cuts = np.genfromtxt("x_cuts_10_worst.csv",delimiter=",")
y_cuts = np.genfromtxt("y_cuts_10_worst.csv",delimiter=",")

boundaries = build_global_subset_boundaries(num_col-1,num_row-1,x_cuts,y_cuts)


for i in range(0,len(boundaries)):
  xmin = boundaries[i][0]
  xmax = boundaries[i][1]
  ymin = boundaries[i][2]
  ymax = boundaries[i][3]
  #Setting up the local for this column.
  pointsx = np.random.uniform(xmin,xmax,light_case)
  pointsy = np.random.uniform(ymin,ymax,light_case)
  points_local = np.stack((pointsx,pointsy),axis=0)
  if i == 0:
    points = points_local
  else:
    points = np.append(points,points_local,axis=1)


plt.figure()
plt.scatter(points.T[:,0],points.T[:,1],s=0.5)
for i in range(0,num_row*num_col):
  
    subset_boundary = boundaries[i]
    xmin = subset_boundary[0]
    xmax = subset_boundary[1]
    ymin = subset_boundary[2]
    ymax = subset_boundary[3]
  
    center_x = (xmin+xmax)/2
    center_y = (ymin+ymax)/2
  
    x = [xmin, xmax, xmax, xmin,xmin]
    y = [ymin, ymin, ymax, ymax,ymin]
  
    plt.plot(x,y,'r')

plt.savefig("../../figures/synthetic_lbd_cuts.pdf")
plt.close()

params = create_parameter_space(x_cuts,y_cuts,num_row,num_col)
num_params = len(params)
bounds = create_bounds(num_params,gxmin,gxmax,gymin,gymax,num_row,num_col)
args = (points,gxmin,gxmax,gymin,gymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)
max_time_lbd = optimized_tts_numerical(params,points,gxmin,gxmax,gymin,gymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)

##max_time = minimize(optimized_tts_numerical,params,method='SLSQP',args=args,bounds=bounds,options={'maxiter':1000,'maxfun':500,'disp':True,'eps':eps},tol=1e-08)
#max_time = optimized_tts_numerical(params,points,gxmin,gxmax,gymin,gymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)
#
x_cuts,y_cuts = create_2d_cuts(gxmin,gxmax,num_col,gymin,gymax,num_row)

params = create_parameter_space(x_cuts,y_cuts,num_row,num_col)
num_params = len(params)
bounds = create_bounds(num_params,gxmin,gxmax,gymin,gymax,num_row,num_col)
args = (points,gxmin,gxmax,gymin,gymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)
params = create_parameter_space(x_cuts,y_cuts,num_row,num_col)
eps = 0.05

max_time = minimize(optimized_tts_numerical,params,method='Newton-CG',jac=jacobian,args=args,bounds=bounds,options={'maxiter':1000,'maxfun':1000,'disp':True},tol=1e-05)
#max_time = differential_evolution(optimized_tts_numerical,bounds,args=args,strategy='best2bin',maxiter=2)
#x_cuts,y_cuts = unpack_parameters(max_time.x,gxmin,gxmax,gymin,gymax,num_col,num_row)
x_cuts,y_cuts = unpack_parameters(max_time.x,gxmin,gxmax,gymin,gymax,num_col,num_row)
boundaries = build_global_subset_boundaries(num_col-1,num_row-1,x_cuts,y_cuts)
plt.figure()
plt.scatter(points.T[:,0],points.T[:,1],s=0.5)
for i in range(0,num_row*num_col):
  
    subset_boundary = boundaries[i]
    xmin = subset_boundary[0]
    xmax = subset_boundary[1]
    ymin = subset_boundary[2]
    ymax = subset_boundary[3]
  
    center_x = (xmin+xmax)/2
    center_y = (ymin+ymax)/2
  
    x = [xmin, xmax, xmax, xmin,xmin]
    y = [ymin, ymin, ymax, ymax,ymin]
  
    plt.plot(x,y,'r')

plt.savefig("../../figures/synthetic_opt_cuts.pdf")
plt.close()



  
  
