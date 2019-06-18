import numpy as np
import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
from sweep_solver import optimized_tts_numerical,unpack_parameters
from mesh_processor import create_2d_cuts
from optimizer import create_parameter_space,create_bounds
from scipy.optimize import minimize,basinhopping
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
lighter_case = 10
heavy_case = 1000

gxmin = 0.0
gxmax = 1.0
gymin = 0.0
gymax = 1.0
num_row = 10
num_col = 10


x_cuts_lbd = np.genfromtxt("x_cuts_10_worst.csv",delimiter=",")
y_cuts_lbd = np.genfromtxt("y_cuts_10_worst.csv",delimiter=",")

boundaries_lbd = build_global_subset_boundaries(num_col-1,num_row-1,x_cuts_lbd,y_cuts_lbd)


for i in range(0,len(boundaries_lbd)):
  xmin = boundaries_lbd[i][0]
  xmax = boundaries_lbd[i][1]
  ymin = boundaries_lbd[i][2]
  ymax = boundaries_lbd[i][3]
  #Setting up the local for this column.
  pointsx = np.random.uniform(xmin,xmax,light_case)
  pointsy = np.random.uniform(ymin,ymax,light_case)
  points_local = np.stack((pointsx,pointsy),axis=0)
  
  pointsx_lighter = np.random.uniform(xmin,xmax,lighter_case)
  pointsy_lighter = np.random.uniform(ymin,ymax,lighter_case)
  points_local_lighter = np.stack((pointsx_lighter,pointsy_lighter),axis=0)
  
  pointsx_heavy = np.random.uniform(xmin,xmax,heavy_case)
  pointsy_heavy = np.random.uniform(ymin,ymax,heavy_case)
  points_local_heavy = np.stack((pointsx_heavy,pointsy_heavy),axis=0)
  
  if i == 0:
    points = points_local
    points_lighter = points_local_lighter
    points_heavy = points_local_heavy
  else:
    points = np.append(points,points_local,axis=1)
    points_lighter = np.append(points_lighter,points_local_lighter,axis=1)
    points_heavy = np.append(points_heavy,points_local_heavy,axis=1)


plt.figure()
plt.scatter(points.T[:,0],points.T[:,1],s=0.5)
for i in range(0,num_row*num_col):
  
    subset_boundary = boundaries_lbd[i]
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

params_lbd = create_parameter_space(x_cuts_lbd,y_cuts_lbd,num_row,num_col)
num_params = len(params_lbd)
max_time_lbd = optimized_tts_numerical(params_lbd,points,gxmin,gxmax,gymin,gymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)

#
x_cuts,y_cuts = create_2d_cuts(gxmin,gxmax,num_col,gymin,gymax,num_row)
params = create_parameter_space(x_cuts,y_cuts,num_row,num_col)
max_time_reg = optimized_tts_numerical(params,points,gxmin,gxmax,gymin,gymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)

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



max_time_lbd_lighter = optimized_tts_numerical(params_lbd,points_lighter,gxmin,gxmax,gymin,gymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)
plt.figure()
plt.scatter(points_lighter.T[:,0],points_lighter.T[:,1],s=0.5)
for i in range(0,num_row*num_col):
  
    subset_boundary = boundaries_lbd[i]
    xmin = subset_boundary[0]
    xmax = subset_boundary[1]
    ymin = subset_boundary[2]
    ymax = subset_boundary[3]
  
    center_x = (xmin+xmax)/2
    center_y = (ymin+ymax)/2
  
    x = [xmin, xmax, xmax, xmin,xmin]
    y = [ymin, ymin, ymax, ymax,ymin]
  
    plt.plot(x,y,'r')

plt.savefig("../../figures/synthetic_lbd_cuts_lighter.pdf")
plt.close()


max_time_reg_lighter = optimized_tts_numerical(params,points_lighter,gxmin,gxmax,gymin,gymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)

plt.figure()
plt.scatter(points_lighter.T[:,0],points_lighter.T[:,1],s=0.5)
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

plt.savefig("../../figures/synthetic_opt_cuts_lighter.pdf")
plt.close()



max_time_lbd_heavy = optimized_tts_numerical(params_lbd,points_heavy,gxmin,gxmax,gymin,gymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)

plt.figure()
plt.scatter(points_heavy.T[:,0],points_heavy.T[:,1],s=0.5)
for i in range(0,num_row*num_col):
  
    subset_boundary = boundaries_lbd[i]
    xmin = subset_boundary[0]
    xmax = subset_boundary[1]
    ymin = subset_boundary[2]
    ymax = subset_boundary[3]
  
    center_x = (xmin+xmax)/2
    center_y = (ymin+ymax)/2
  
    x = [xmin, xmax, xmax, xmin,xmin]
    y = [ymin, ymin, ymax, ymax,ymin]
  
    plt.plot(x,y,'r')

plt.savefig("../../figures/synthetic_lbd_cuts_heavy.png")
plt.close()

max_time_reg_heavy = optimized_tts_numerical(params,points_heavy,gxmin,gxmax,gymin,gymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)

  
plt.figure()
plt.scatter(points_heavy.T[:,0],points_heavy.T[:,1],s=0.5)
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

plt.savefig("../../figures/synthetic_opt_cuts_heavy.png")
plt.close()
