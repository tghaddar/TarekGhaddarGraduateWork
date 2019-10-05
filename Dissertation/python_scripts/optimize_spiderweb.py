import numpy as np
import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
from sweep_solver import optimized_tts_numerical,unpack_parameters,plot_subset_boundaries_2d
from mesh_processor import create_2d_cuts
from optimizer import create_parameter_space,create_bounds,create_constraints,get_column_cdf,create_opt_cut_suite,get_highest_jumps
from scipy.optimize import basinhopping, minimize
import matplotlib.pyplot as plt
from build_global_subset_boundaries import build_global_subset_boundaries
plt.close("all")
plt.close("all")
#Communication time per double
t_comm = 4.47e-09
#The number of bytes to communicate per subset.
#The message latency time.
m_l = 50
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

f = open("sparse_pins_cell_verts",'r')
sparse_pins_cell_data = [line.split() for line in f]
for i in range(0,len(sparse_pins_cell_data)):
  sparse_pins_cell_data[i] = [int(x) for x in sparse_pins_cell_data[i]]

verts = np.genfromtxt("unbalanced_pins_sparse_vert_data")


gxmin = 0.0
gxmax = 10.0
gymin = 0.0
gymax = 10.0
add_cells = True

numrow = 3
numcol = 3
x_cuts,y_cuts = create_2d_cuts(gxmin,gxmax,numcol,gymin,gymax,numrow)
params = create_parameter_space(x_cuts,y_cuts,numrow,numcol)
num_params = len(params)

bounds = create_bounds(num_params,gxmin,gxmax,gymin,gymax,numrow,numcol)
constraints = create_constraints(gxmin,gxmax,gymin,gymax,numrow,numcol)
args = (sparse_pins_cell_data,verts,gxmin,gxmax,gymin,gymax,numrow,numcol,machine_parameters,num_angles,Am,Ay,add_cells,unweighted)
#max_time = basinhopping(optimized_tts_numerical,params,niter=1000,stepsize=0.5,minimizer_kwargs={"method":"Nelder-Mead","bounds":bounds,"constraints":constraints,'args':args,'options':{'maxiter':100}})
#max_time = minimize(optimized_tts_numerical,params,args=args,bounds=bounds,constraints=constraints,method='SLSQP',options={"maxiter":1000})