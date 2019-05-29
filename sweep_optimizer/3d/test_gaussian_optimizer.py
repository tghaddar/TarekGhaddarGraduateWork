import numpy as np
import warnings
import matplotlib.pyplot as plt
from mesh_processor import create_2d_cuts,get_cells_per_subset_2d,create_2d_cut_suite
from optimizer import create_parameter_space,create_bounds
from sweep_solver import optimized_tts_numerical,time_to_solution_numerical,optimized_tts
from sweep_solver import time_to_solution
from scipy.optimize import minimize
import time
warnings.filterwarnings("ignore")

plt.close("all")

mean = [0.5,0.5]
cov = [[0.01,0],[0,0.01]]

x,y = np.random.multivariate_normal(mean,cov,100).T
points = [x,y]

f = lambda x,y: x + y

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

#Number of rows and columns.
numrow = 2
numcol = 2
num_angles = 1

#Global boundaries.
global_xmin = 0.0
global_xmax = 10.0
global_ymin = 0.0
global_ymax = 10.0

x_cuts,y_cuts = create_2d_cuts(global_xmin,global_xmax,numcol,global_ymin,global_ymax,numrow)
all_x_cuts,all_y_cuts = create_2d_cut_suite(global_xmin,global_xmax,numcol,global_ymin,global_ymax,numrow)

num_x_cuts = len(all_x_cuts)
num_y_cuts = len(all_y_cuts)
max_times = []



interior_cuts = create_parameter_space(x_cuts,y_cuts,numrow,numcol)
num_params = len(interior_cuts)
bounds = create_bounds(num_params,global_xmin,global_xmax,global_ymin,global_ymax,numrow,numcol)
args = (f,global_xmin,global_xmax,global_ymin,global_ymax,numrow,numcol,t_u,upc,upbc,t_comm,latency,m_l,num_angles)
start = time.time()
max_time = minimize(optimized_tts,interior_cuts,method='Nelder-Mead',args = args,bounds = bounds,options={'maxiter':1000,'maxfun':1000,'disp':False},tol=1e-08)
end = time.time()
print(end - start)

#x_cuts = [0.0, 6.145603174757288, 10.0] 
#y_cuts = [[0.0, 5.0, 10.0], [0.0, 6.145603174757288, 10.0]]
#max_time = time_to_solution(f,x_cuts,y_cuts,machine_params,numcol,numrow,num_angles)
#max_time = time_to_solution_numerical(points,all_x_cuts[0],all_y_cuts[0],machine_params,numcol,numrow)
#for i in range(0,num_x_cuts):
#  for j in range(0,num_y_cuts): 
#    print(i,j)
#    x_cuts = all_x_cuts[i]
#    y_cuts = all_y_cuts[j]    
#    x_cut = x_cuts[1]
#    y_cut_0 = y_cuts[0][1]
#    y_cut_1 = y_cuts[1][1]
#    max_time = time_to_solution_numerical(points,x_cuts,y_cuts,machine_params,numcol,numrow)
#    max_times.append([x_cut,y_cut_0,y_cut_1,max_time])
#    print("here")
