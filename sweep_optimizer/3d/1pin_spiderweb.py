import numpy as np
from optimizer import create_parameter_space,create_bounds
from mesh_processor import create_2d_cuts
import sweep_solver as ss
from scipy.optimize import minimize

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

points = np.genfromtxt("1pin_centroids")
points = points.T

global_xmin = 0.0
global_xmax = 0.25
global_ymin = 0.0
global_ymax = 0.25

numcol = 3
numrow = 3
num_angles = 1
unweighted = True
test = False

x_cuts,y_cuts = create_2d_cuts(global_xmin,global_xmax,numcol,global_ymin,global_ymax,numrow)
interior_cuts = create_parameter_space(x_cuts,y_cuts,numrow,numcol)
num_params = len(interior_cuts)

bounds = create_bounds(num_params,global_xmin,global_xmax,global_ymin,global_ymax,numrow,numcol)
args = (points,global_xmin,global_xmax,global_ymin,global_ymax,numrow,numcol,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)

max_time = minimize(ss.optimized_tts_numerical,interior_cuts,method='SLSQP',args = args,bounds = bounds,options={'maxiter':1000,'maxfun':1000,'disp':False},tol=1e-08)

final_params = max_time.x
x_cuts,y_cuts = ss.unpack_parameters(final_params,global_xmin,global_xmax,global_ymin,global_ymax,numcol,numrow)
#Writing the xml portions.
f = open("cuts.xml",'w')
f.write("<x_cuts>")
for x in range(1,numcol):
  f.write(str(x_cuts[x])+" ")

f.write("</x_cuts>\n")

f.write("<y_cuts_by_column>\n")
for col in range(0,numcol):
  f.write("  <column>")
  for y in range(1,numrow):
    f.write(str(y_cuts[col][y])+ " ")
  
  f.write("</column>\n")

f.write("</y_cuts_by_column>\n")
f.close()