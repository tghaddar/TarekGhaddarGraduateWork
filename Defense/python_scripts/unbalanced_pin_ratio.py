import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
#sys.path.append(r'C:\Users\tghad\Documents\GitHub\TarekGhaddarGraduateWork\sweep_optimizer\3d')
from sweep_solver import optimized_tts_numerical
from optimizer import create_parameter_space
from mesh_processor import create_2d_cuts
import uncertainties.unumpy as unumpy  

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



max_time_reg_pdt = np.genfromtxt("solvepersweep_regular_fcfs.txt")
max_time_lb_pdt = np.genfromtxt("solvepersweep_lb_fcfs.txt")
max_time_reg_shape = np.reshape(max_time_reg_pdt,(10,9))
max_time_lb_shape = np.reshape(max_time_lb_pdt,(10,9))
max_time_reg_median = np.zeros([9])
max_time_lb_median = np.zeros([9])
max_time_reg_stdev = np.zeros([9])
max_time_lb_stdev = np.zeros([9])
for i in range(0,9):
  cur_med = np.median(max_time_reg_shape[:,i])
  stdev = np.std(max_time_reg_shape[:,i])
  cur_med_lb = np.median(max_time_lb_shape[:,i])
  stdev_lb = np.std(max_time_lb_shape[:,i])
  max_time_reg_median[i] = cur_med
  max_time_lb_median[i] = cur_med_lb
  max_time_reg_stdev[i] = stdev
  max_time_lb_stdev[i] = stdev_lb
  
x = unumpy.uarray((max_time_reg_median,max_time_reg_stdev))
y = unumpy.uarray((max_time_lb_median,max_time_lb_stdev))
ratio_pdt_u = x/y
ratio_tts_std = np.zeros([9])
ratio_tts_u = unumpy.uarray((ratio_tts,ratio_tts_std))
#ratio_pdt = np.divide(max_time_lb_median,max_time_reg_median)
#
diff = abs(ratio_pdt_u-ratio_tts_u)
percent_diff = diff/ratio_pdt_u

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