import numpy as np
from mesh_processor import create_2d_cuts
from optimizer import create_parameter_space, create_bounds, create_constraints
from sweep_solver import optimized_tts
from scipy.optimize import minimize,basinhopping


f = lambda x,y: x + y
#The machine parameters.
#Communication time per double
t_comm = 4.47e-05
#The number of bytes to communicate per subset.
#The message latency time.
m_l = 1
latency = 4110.0e-05
#Solve time per unknown.
t_u = 450.0e-05
upc = 4.0
upbc = 2.0

#Number of rows and columns.
numrow = 3
numcol = 3
num_angles = 1
unweighted=True

#Global boundaries.
global_xmin = 0.0
global_xmax = 10.0
global_ymin = 0.0
global_ymax = 10.0


x_cuts,y_cuts = create_2d_cuts(global_xmin,global_xmax,numcol,global_ymin,global_ymax,numrow)
interior_cuts = create_parameter_space(x_cuts,y_cuts,numrow,numcol)
num_params = len(interior_cuts)

bounds = create_bounds(num_params,global_xmin,global_xmax,global_ymin,global_ymax,numrow,numcol)
constraints = create_constraints(global_xmin,global_xmax,global_ymin,global_ymax,numrow,numcol)
args = (f,global_xmin,global_xmax,global_ymin,global_ymax,numrow,numcol,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)

max_time = basinhopping(optimized_tts,interior_cuts,minimizer_kwargs={'method':'SLSQP','bounds':bounds,'constraints':constraints,'args':args})

