
import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
import numpy as np
from sweep_solver import optimized_tts_numerical,unpack_parameters
from mesh_processor import create_2d_cuts
from optimizer import create_parameter_space,create_bounds
from scipy.optimize import minimize,basinhopping

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

ns = [2]
num_suite = len(ns)

num_angles = 1
unweighted = True
xmin = 0.0
xmax = 10.0
ymin = 0.0
ymax = 10.0
points = np.genfromtxt("unbalanced_pins_sparse_centroid_data").T
all_x_cuts = [None]*num_suite
all_y_cuts = [None]*num_suite

for i in range(0,num_suite):
  s = ns[i]
  print(s)
  num_row = s
  num_col = s
  
#  x_cuts,y_cuts = create_2d_cuts(xmin,xmax,num_col,ymin,ymax,num_row)
  x_cuts=[0.0,5.0,10.0]
  y_cuts = [[0.0,1.18519,10.0],[0.0,8.81481,10.0]]
#  x_cuts = [0.0, 0.25, 0.5000000149011612, 0.75, 1.0] 
#  y_cuts = [[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0]]
  params = create_parameter_space(x_cuts,y_cuts,num_row,num_col)
  num_params = len(params)
  bounds = create_bounds(num_params,xmin,xmax,ymin,ymax,num_row,num_col)
  args = (points,xmin,xmax,ymin,ymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)
#  x_cuts_lbd = [xmin,]
  #max_time = (optimized_tts_numerical,params,method='SLSQP',args=args,bounds=bounds,options={'maxiter':1000,'maxfun':1000,'disp':True},tol=1e-08)
#  mt = optimized_tts_numerical(params,points,xmin,xmax,ymin,ymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)
  max_time = basinhopping(optimized_tts_numerical,params,niter=10,T=1.0,stepsize=0.1,minimizer_kwargs={"args":args,"bounds":bounds,"method":"SLSQP","options":{'eps':0.01}})
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

num_row = 2
num_col = 2  
x_cuts_lbd=[0.0,5.0,10.0]
y_cuts_lbd = [[0.0,1.18519,10.0],[0.0,8.81481,10.0]]
params = create_parameter_space(x_cuts_lbd,y_cuts_lbd,num_row,num_col)
num_params = len(params)
bounds = create_bounds(num_params,xmin,xmax,ymin,ymax,num_row,num_col)
args = (points,xmin,xmax,ymin,ymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)
mt = optimized_tts_numerical(params,points,xmin,xmax,ymin,ymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)
