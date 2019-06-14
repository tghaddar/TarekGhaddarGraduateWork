
import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
import numpy as np
from sweep_solver import optimized_tts_numerical,unpack_parameters
from mesh_processor import create_2d_cuts
from optimizer import create_parameter_space,create_bounds
from scipy.optimize import minimize

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

ns = [2,3,4,5,6,8,9]
num_suite = len(ns)

num_angles = 1
unweighted = True
xmin = 0.0
xmax = 1.0
ymin = 0.0
ymax = 1.0
points = np.genfromtxt("unbalanced_pins_centroid_data").T
all_x_cuts = [None]*num_suite
all_y_cuts = [None]*num_suite

for i in range(0,num_suite):
  s = ns[i]
  print(s)
  num_row = s
  num_col = s
  
  x_cuts,y_cuts = create_2d_cuts(xmin,xmax,num_col,ymin,ymax,num_row)
  params = create_parameter_space(x_cuts,y_cuts,num_row,num_col)
  num_params = len(params)
  bounds = create_bounds(num_params,xmin,xmax,ymin,ymax,num_row,num_col)
  args = (points,xmin,xmax,ymin,ymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)
  max_time = minimize(optimized_tts_numerical,params,method='SLSQP',args=args,bounds=bounds,options={'maxiter':1000,'maxfun':1000,'disp':False},tol=1e-08)
  x_cuts,y_cuts = unpack_parameters(max_time.x,xmin,xmax,ymin,ymax,num_col,num_row)
  print(max_time)
  all_x_cuts[i] = x_cuts
  all_y_cuts[i] = y_cuts
  f = open("cuts_"+str(s)+".xml",'w')
  f.write("<x_cuts>")
  for x in range(1,num_col):
    f.write(str(x_cuts[x])+" ")
    
  f.write("</x_cuts>\n")
  f.write("<y_cuts_by_column>\n")
  for col in range(0,num_col):
    f.write("  <column>")
    for y in range(1,num_row):
      f.write(str(y_cuts[col][y])+ " ")
    
    f.write("</column>\n")

  f.write("</y_cuts_by_column>\n")
  f.close()