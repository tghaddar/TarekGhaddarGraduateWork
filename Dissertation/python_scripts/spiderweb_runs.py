import numpy as np
import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
from sweep_solver import optimized_tts_numerical,unpack_parameters
from mesh_processor import create_2d_cuts
from optimizer import create_parameter_space,create_bounds,create_constraints,get_column_cdf,create_opt_cut_suite,get_highest_jumps
from scipy.optimize import basinhopping, minimize
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
plt.close("all")
#Communication time per double
t_comm = 4.47e-09
#The number of bytes to communicate per subset.
#The message latency time.
m_l = 1
latency = 4110.0e-09
#Solve time per cell..
Tc = 1208.383e-09
upc = 4.0
upbc = 2.0
Twu = 147.0754e-09
Tm = 65.54614e-09
Tg = 175.0272e-09
mcff = 1.181
machine_parameters = (Twu,Tc,Tm,Tg,upc,upbc,mcff,t_comm,latency,m_l)
num_angles = 1
Am = 36
unweighted = True
Ay = 1

points = np.genfromtxt("unbalanced_pins_sparse_centroid_data").T
verts = np.genfromtxt("unbalanced_pins_sparse_vert_data")

gxmin = 0.0
gxmax = 10.0
gymin = 0.0
gymax = 10.0

numrows = [1,2,3,4,5,6,7,8,9,10]
numcols = [1,2,3,4,5,6,7,8,9,10]

x_values = get_highest_jumps(verts[:,0],gxmin,gxmax,10)

#max_times_case = {}
#
#for i in range(0,len(numrows)):
#  numrow = numrows[i]
#  numcol = numcols[i]
#  
#  #y_values_func = get_highest_jumps(verts[:,1],gymin,gymax,numcol)
#  x_values,y_cut_suite = create_opt_cut_suite(verts,gxmin,gxmax,gymin,gymax,numcol,numrow)
#  
#  max_times = []
#  add_cells = False
#  for j in range(0,len(y_cut_suite)):
#    x_cuts = x_values
#    y_cuts = y_cut_suite[j]
#    params = create_parameter_space(x_cuts,y_cuts,numrow,numcol)
#    max_times.append(optimized_tts_numerical(params,points,gxmin,gxmax,gymin,gymax,numrow,numcol,machine_parameters,num_angles,Am,Ay,add_cells,unweighted))
#  
#  min_index = max_times.index(min(max_times))
#  y_cuts_min = y_cut_suite[min_index]
#  x_cuts_min = x_values
#  
#  max_times_case[numrow] = (min(max_times),x_cuts_min,y_cuts_min)




#Trying optimizing the spiderweb.
#x_cuts,y_cuts = create_2d_cuts(gxmin,gxmax,numcol,gymin,gymax,numrow)
#
#params = create_parameter_space(x_cuts,y_cuts,numrow,numcol)
#num_params = len(params)
#
#bounds = create_bounds(num_params,gxmin,gxmax,gymin,gymax,numrow,numcol)
#constraints = create_constraints(gxmin,gxmax,gymin,gymax,numrow,numcol)
#args = (points,gxmin,gxmax,gymin,gymax,numrow,numcol,machine_parameters,num_angles,Am,Ay,add_cells,unweighted)
#max_time = basinhopping(optimized_tts_numerical,params,niter=200,stepsize=0.5,minimizer_kwargs={"method":"Nelder-Mead","bounds":bounds,"constraints":constraints,'args':args,'options':{'maxiter':1}})