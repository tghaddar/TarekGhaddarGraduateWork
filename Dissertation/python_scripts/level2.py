import numpy as np
import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
from sweep_solver import optimized_tts_numerical,unpack_parameters
from mesh_processor import create_2d_cuts
from optimizer import create_parameter_space,create_bounds,create_constraints,get_column_cdf,create_opt_cut_suite,get_highest_jumps
from scipy.optimize import basinhopping, minimize
import matplotlib.pyplot as plt
import itertools
plt.close("all")

gxmin = 0.0
gxmax = 152.0
gymin = 0.0
gymax = 54.994

#Communication time per double
t_comm = 4.47e-09
#The number of bytes to communicate per subset.
#The message latency time.
m_l = 35
latency = 4110.0e-09
#Solve time per cell..
Tc = 1208.383e-09
upc = 4.0
upbc = 2.0
Twu = 147.0754e-09
Tm = 65.54614e-09
Tg = 175.0272e-09
mcff = 1.32
machine_parameters = (Twu,Tc,Tm,Tg,upc,upbc,mcff,t_comm,latency,m_l)
num_angles = 1
Am = 36
unweighted = True
Ay = 1
numcol = 42
numrow = 13

points = np.genfromtxt("level2centroids").T

#Trying optimizing the spiderweb.
x_cuts,y_cuts = create_2d_cuts(gxmin,gxmax,42,gymin,gymax,13)
params = create_parameter_space(x_cuts,y_cuts,13,42)

#max_time_reg = optimized_tts_numerical(params,points,gxmin,gxmax,gymin,gymax,13,42,machine_parameters,num_angles,Am,Ay,unweighted)


x_cuts_lb = [0.0,7.0,14.62,16.1565,17.16,18.1635,19.7,30.5,38.76,47.9,55.52,64.66,67.835,68.47,69.105,69.74,71.53,71.78,72.03,72.28,73.27,74.26,74.92,75.58,76.24,76.9,77.89,78.88,79.13,79.38,79.63,81.42,82.055,82.69,83.325,86.5,95.64,103.26,112.4,120.66,130.88,141.44,gxmax]
y_cuts_lbd_col = [0.0,19.1775,31.228,43.8345,47.0373,48.0957,48.7307,49.7507,51.194,51.5273,52.024,53.014,54.04,54.994]
y_cuts_lb = []
for col in range(0,numcol):
  y_cuts_lb.append(y_cuts_lbd_col)

params = create_parameter_space(x_cuts_lb,y_cuts_lb,numrow,numcol)
num_params=len(params)
add_cells = True
#max_time_lb = optimized_tts_numerical(params,points,gxmin,gxmax,gymin,gymax,numrow,numcol,machine_parameters,num_angles,Am,Ay,add_cells,unweighted)

#bounds = create_bounds(num_params,gxmin,gxmax,gymin,gymax,13,42)
#constraints = create_constraints(gxmin,gxmax,gymin,gymax,13,42)
#args = (points,gxmin,gxmax,gymin,gymax,13,42,machine_parameters,num_angles,Am,Ay,unweighted)
##max_time = minimize(optimized_tts_numerical,params,args=args,bounds=bounds,constraints=constraints,method='COBYLA',options={"maxiter":1})
#max_time = basinhopping(optimized_tts_numerical,params,niter=200,stepsize=0.5,minimizer_kwargs={"method":"COBYLA","bounds":bounds,"constraints":constraints,'args':args,'options':{'maxiter':1}})
##print(max_time_reg,max_time_lb)
verts = np.genfromtxt("level2_vert_data")
x_values = get_highest_jumps(verts[:,0],gxmin,gxmax,numcol)


x_values,y_cut_suite = create_opt_cut_suite(verts,gxmin,gxmax,gymin,gymax,numcol,numrow)


#max_times = []
#add_cells = False
#for i in range(0,len(y_cut_suite)):
#  x_cuts = x_values
#  y_cuts = y_cut_suite[i]
#  params = create_parameter_space(x_cuts,y_cuts,numrow,numcol)
#  max_times.append(optimized_tts_numerical(params,points,gxmin,gxmax,gymin,gymax,numrow,numcol,machine_parameters,num_angles,Am,Ay,add_cells,unweighted))
#
#min_index = max_times.index(min(max_times))
#y_cuts_min = y_cut_suite[min_index]
#x_cuts_min = x_values
#
##Writing the xml portions.
#f = open("level2_opt_cuts.xml",'w')
#f.write("<x_cuts>")
#for x in range(1,numcol):
#  f.write(str(x_cuts_min[x])+" ")
#
#f.write("</x_cuts>\n")
#
#f.write("<y_cuts_by_column>\n")
#for col in range(0,numcol):
#  f.write("  <column>")
#  for y in range(1,numrow):
#    f.write(str(y_cuts_min[col][y])+ " ")
#  
#  f.write("</column>\n")
#
#f.write("</y_cuts_by_column>\n")
#f.close()
