import numpy as np
import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
from sweep_solver import optimized_tts_numerical,unpack_parameters
from mesh_processor import create_2d_cuts
from optimizer import create_parameter_space,create_bounds
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt
plt.close("all")

gxmin = 0.0
gxmax = 152.0
gymin = 0.0
gymax = 54.994

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


points = np.genfromtxt("level2centroids").T

num_subsets = [2,3,4,5,6,7,8,9,10]
num_runs = len(num_subsets)
max_times = [None]*num_runs
ctr = 0
for n in num_subsets:
  num_row = n
  num_col = n
  
  x_cuts,y_cuts = create_2d_cuts(gxmin,gxmax,num_row,gymin,gymax,num_col)
  
  params = create_parameter_space(x_cuts,y_cuts,num_row,num_col)
  max_time = optimized_tts_numerical(params,points,gxmin,gxmax,gymin,gymax,num_row,num_col,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)
  max_times[ctr] = max_time
  ctr+=1
  
plt.figure()
plt.title("Level 2 Solution Times against Number of Subsets in Each Dimension")
plt.xlabel("Number of Subsets in Each Dimension")
plt.ylabel("Solution Time")
plt.scatter(num_subsets,max_times)
plt.savefig("../../figures/level2solutionsuite.pdf")

#Trying optimizing the spiderweb.
x_cuts,y_cuts = create_2d_cuts(gxmin,gxmax,42,gymin,gymax,13)
params = create_parameter_space(x_cuts,y_cuts,13,42)
#num_params = len(params)
#bounds = create_bounds(num_params,gxmin,gxmax,gymin,gymax,2,2)
#args = (points,gxmin,gxmax,gymin,gymax,2,2,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)
#max_time = basinhopping(optimized_tts_numerical,params,niter=10,stepsize=5.0,minimizer_kwargs={"args":args,"bounds":bounds,"method":"SLSQP","tol":1e-03,"options":{'eps':0.5}})

max_time_reg = optimized_tts_numerical(params,points,gxmin,gxmax,gymin,gymax,13,42,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)


x_cuts_lb = [0.0,7.0,14.62,16.1565,17.16,18.1635,19.7,30.5,38.76,47.9,55.52,64.66,67.835,68.47,69.105,69.74,71.53,71.78,72.03,72.28,73.27,74.26,74.92,75.58,76.24,76.9,77.89,78.88,79.13,79.38,79.63,81.42,82.055,82.69,83.325,86.5,95.64,103.26,112.4,120.66,130.88,141.44,gxmax]
y_cuts_lbd_col = [0.0,19.1775,31.228,43.8345,47.0373,48.0957,48.7307,49.7507,51.194,51.5273,52.024,53.014,54.04,54.994]
y_cuts_lb = []
for col in range(0,42):
  y_cuts_lb.append(y_cuts_lbd_col)

params = create_parameter_space(x_cuts_lb,y_cuts_lb,13,42)
max_time_lb = optimized_tts_numerical(params,points,gxmin,gxmax,gymin,gymax,13,42,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted)