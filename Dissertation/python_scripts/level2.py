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
m_l = 40
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

points = np.genfromtxt("level2centroids").T

#Trying optimizing the spiderweb.
x_cuts,y_cuts = create_2d_cuts(gxmin,gxmax,42,gymin,gymax,13)
params = create_parameter_space(x_cuts,y_cuts,13,42)

max_time_reg = optimized_tts_numerical(params,points,gxmin,gxmax,gymin,gymax,13,42,machine_parameters,num_angles,Am,Ay,unweighted)


x_cuts_lb = [0.0,7.0,14.62,16.1565,17.16,18.1635,19.7,30.5,38.76,47.9,55.52,64.66,67.835,68.47,69.105,69.74,71.53,71.78,72.03,72.28,73.27,74.26,74.92,75.58,76.24,76.9,77.89,78.88,79.13,79.38,79.63,81.42,82.055,82.69,83.325,86.5,95.64,103.26,112.4,120.66,130.88,141.44,gxmax]
y_cuts_lbd_col = [0.0,19.1775,31.228,43.8345,47.0373,48.0957,48.7307,49.7507,51.194,51.5273,52.024,53.014,54.04,54.994]
y_cuts_lb = []
for col in range(0,42):
  y_cuts_lb.append(y_cuts_lbd_col)

params = create_parameter_space(x_cuts_lb,y_cuts_lb,13,42)
max_time_lb = optimized_tts_numerical(params,points,gxmin,gxmax,gymin,gymax,13,42,machine_parameters,num_angles,Am,Ay,unweighted)

print(max_time_reg,max_time_lb)