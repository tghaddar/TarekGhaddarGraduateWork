import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
#sys.path.append(r'C:\Users\tghad\Documents\GitHub\TarekGhaddarGraduateWork\sweep_optimizer\3d')
from sweep_solver import optimized_tts_numerical
from optimizer import create_parameter_space
from mesh_processor import create_2d_cuts


#The machine parameters.
#Communication time per double
t_comm = 4.47e-02
#The number of bytes to communicate per subset.
#The message latency time.
m_l = 1
latency = 4110.0e-02
#Solve time per unknown.
t_u = 450.0e-02
upc = 4.0
upbc = 2.0
machine_params = (t_u,upc,upbc,t_comm,latency,m_l)

num_row = 3
num_col = 3
num_angles = 1
unweighted = True
xmin = 0.0
xmax = 1.0
ymin = 0.0
ymax = 1.0

x_cuts,y_cuts = create_2d_cuts(xmin,xmax,num_col,ymin,ymax,num_row)
points = np.genfromtxt("unbalanced_pins_centroid_data").T
params = create_parameter_space(x_cuts,y_cuts,num_row,num_col)

max_time_reg = optimized_tts_numerical(params, points,xmin,xmax,ymin,ymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)


#LBD cuts.
x_cuts = np.genfromtxt("cut_line_data_x")
y_cuts = np.genfromtxt("cut_line_data_y")

params = create_parameter_space(x_cuts,y_cuts,num_row,num_col)

max_time_lbd = optimized_tts_numerical(params, points,xmin,xmax,ymin,ymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)

improvement = max_time_lbd/max_time_reg

#Writing the xml portions.
f = open("cuts.xml",'w')
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