import numpy as np

import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
#sys.path.append(r'C:\Users\tghad\Documents\GitHub\TarekGhaddarGraduateWork\sweep_optimizer\3d')
from sweep_solver import optimized_tts_numerical
from optimizer import create_parameter_space,create_bounds
from mesh_processor import create_2d_cuts

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
xmin = 0.0
xmax = 10.0
ymin = 0.0
ymax = 10.0

ns = [2,3,4,6]
num_suite = len(ns)
ratio_tts_lbd = [0.0]*(num_suite)

for i in range(0,num_suite):
  s = ns[i]
  num_row = s
  num_col = s
  
  x_cuts,y_cuts = create_2d_cuts(xmin,xmax,num_col,ymin,ymax,num_row)
  points = np.genfromtxt("unbalanced_pins_sparse_centroid_data").T
  params = create_parameter_space(x_cuts,y_cuts,num_row,num_col)
  
  max_time_reg = optimized_tts_numerical(params, points,xmin,xmax,ymin,ymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)
  
  
  #LB cuts.
  x_cuts = np.genfromtxt("cut_line_data_x_"+str(s)+"_lbd_sparse")
  y_cuts = np.genfromtxt("cut_line_data_y_"+str(s)+"_lbd_sparse")
  
  params = create_parameter_space(x_cuts,y_cuts,num_row,num_col)
  
  max_time_lbd = optimized_tts_numerical(params, points,xmin,xmax,ymin,ymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)
  
  ratio_tts_lbd[i] = max_time_lbd/max_time_reg

max_time_reg_pdt = np.genfromtxt("solvepersweep_sparse_regular_fcfs.txt")
max_time_lbd_pdt = np.genfromtxt("solvepersweep_sparse_lbd_fcfs.txt")
max_time_reg_shape = np.reshape(max_time_reg_pdt,(10,9))
max_time_reg_shape = max_time_reg_shape[:,[0,1,2,4]]
max_time_lbd_shape = np.reshape(max_time_lbd_pdt,(10,4))
max_time_reg_median = np.zeros([4])
max_time_lbd_median = np.ones([4])
for i in range(0,4):
  s = ns[i]
  max_time_reg_median[i] = np.median(max_time_reg_shape[:,i])
  max_time_lbd_median[i] = np.median(max_time_lbd_shape[:,i])

  

ratio_pdt_lbd = np.divide(max_time_lbd_median,max_time_reg_median)
np.savetxt("ratio_pdt_sparse_lbd",ratio_pdt_lbd)
np.savetxt("ratio_tts_sparse_lbd",ratio_tts_lbd)

diff_lbd = abs(ratio_pdt_lbd-ratio_tts_lbd)
percent_diff_lbd = diff_lbd/ratio_pdt_lbd

##Writing the xml portions.
#f = open("cuts.xml",'w')
#f.write("<x_cuts>")
#for x in range(1,num_col):
#  f.write(str(x_cuts[x])+" ")
#
#f.write("</x_cuts>\n")
#
#f.write("<y_cuts_by_column>\n")
#for col in range(0,num_col):
#  f.write("  <column>")
#  for y in range(1,num_row):
#    f.write(str(y_cuts[col][y])+ " ")
#  
#  f.write("</column>\n")
#
#f.write("</y_cuts_by_column>\n")
#f.close()
