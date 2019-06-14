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
xmax = 1.0
ymin = 0.0
ymax = 1.0

ns = [2,3,4,5,6,7,8,9,10]
num_suite = len(ns)
ratio_tts = [None]*num_suite
ratio_tts_lbd = [0.0]*(num_suite-1)

for i in range(0,num_suite):
  s = ns[i]
  num_row = s
  num_col = s
  
  x_cuts,y_cuts = create_2d_cuts(xmin,xmax,num_col,ymin,ymax,num_row)
  points = np.genfromtxt("unbalanced_pins_centroid_data").T
  params = create_parameter_space(x_cuts,y_cuts,num_row,num_col)
  
  max_time_reg = optimized_tts_numerical(params, points,xmin,xmax,ymin,ymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)
  
  
  #LB cuts.
  x_cuts = np.genfromtxt("cut_line_data_x_"+str(s))
  y_cuts = np.genfromtxt("cut_line_data_y_"+str(s))
  
  params = create_parameter_space(x_cuts,y_cuts,num_row,num_col)
  
  max_time_lb = optimized_tts_numerical(params, points,xmin,xmax,ymin,ymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)
  
  ratio_tts[i] = max_time_lb/max_time_reg
  
  #LBD
  if (i < num_suite-1) and (s!=7):
    x_cuts = np.genfromtxt("cut_line_data_x_"+str(s)+"_lbd")
    y_cuts = np.genfromtxt("cut_line_data_y_"+str(s)+"_lbd") 
    max_time_lbd = optimized_tts_numerical(params, points,xmin,xmax,ymin,ymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)
    ratio_tts_lbd[i] = max_time_lbd/max_time_reg

max_time_reg_pdt = np.genfromtxt("solvepersweep_regular_fcfs.txt")
max_time_lb_pdt = np.genfromtxt("solvepersweep_lb_fcfs.txt")
max_time_lbd_pdt = np.genfromtxt("solvepersweep_lbd_fcfs.txt")
max_time_reg_shape = np.reshape(max_time_reg_pdt,(10,9))
max_time_lb_shape = np.reshape(max_time_lb_pdt,(10,9))
max_time_lbd_shape = np.reshape(max_time_lbd_pdt,(10,7))
max_time_lbd_shape = np.insert(max_time_lbd_shape,5,np.zeros(10).T,axis=1)
max_time_reg_median = np.zeros([9])
max_time_lb_median = np.zeros([9])
max_time_lbd_median = np.ones([8])
max_time_reg_stdev = np.zeros([9])
max_time_lb_stdev = np.zeros([9])
for i in range(0,9):
  s = ns[i]
  cur_med = np.median(max_time_reg_shape[:,i])
  stdev = np.std(max_time_reg_shape[:,i])
  cur_med_lb = np.median(max_time_lb_shape[:,i])
  stdev_lb = np.std(max_time_lb_shape[:,i])
  max_time_reg_median[i] = cur_med
  max_time_lb_median[i] = cur_med_lb
  max_time_reg_stdev[i] = stdev
  max_time_lb_stdev[i] = stdev_lb
  
  if (i < 8) and (s != 7):
    max_time_lbd_median[i] = np.median(max_time_lbd_shape[:,i])

  

ratio_pdt_lb = np.divide(max_time_lb_median,max_time_reg_median)
ratio_pdt_lbd = np.divide(max_time_lbd_median,max_time_reg_median[0:8])
np.savetxt("ratio_pdt_lb",ratio_pdt_lb)
np.savetxt("ratio_tts_lb",ratio_tts)
np.savetxt("ratio_pdt_lbd",ratio_pdt_lbd)
np.savetxt("ratio_tts_lbd",ratio_tts_lbd)

#
diff_lb = abs(ratio_pdt_lb-ratio_tts)
percent_diff_lb = diff_lb/ratio_pdt_lb
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