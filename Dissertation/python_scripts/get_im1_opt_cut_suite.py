import sys
sys.path.append(r'C:\Users\tghad\Documents\GitHub\TarekGhaddarGraduateWork\sweep_optimizer\3d')
from optimizer import create_parameter_space_3d,create_opt_cut_suite_3d,create_opt_cut_suite_3d_given_z
import numpy as np
import sweep_solver as ss


#Communication time per double
t_comm = 4.47e-09
#The number of bytes to communicate per subset.
#The message latency time.
m_l = 2.5
latency = 4110.0e-09
#Solve time per cell.
Tc = 2683.769129e-09
upc = 8.0
upbc = 4.0
Twu = 5779.92891144936e-09
Tm = 111.971932555903e-09
Tg = 559.127e-09
mcff = 1.32

machine_parameters = (Twu,Tc,Tm,Tg,upc,upbc,mcff,t_comm,latency,m_l)
Am = 10
Az = 1
num_angles = 1
unweighted=True
test = False
add_cells = True

gxmin = 0.0
gxmax = 60.96
gymin = 0.0
gymax = 60.96
gzmin = 0.0
gzmax = 146.05
numrow = 5
numcol = 5
numplane = 5
im1points = np.genfromtxt("im1_cell_centers")
points = np.genfromtxt("im1_points")
z_cuts = np.genfromtxt("im1_lbd_z")
x_cut_suite,y_cut_suite = create_opt_cut_suite_3d_given_z(points,z_cuts,gxmin,gxmax,gymin,gymax,gzmin,gzmax,numcol,numrow,numplane)

max_times = []
for i in range(0,len(x_cut_suite)):
  for j in range(0,len(y_cut_suite)):
    x_cuts = x_cut_suite[i]
    y_cuts = y_cut_suite[j]
    add_cells = False
    params = create_parameter_space_3d(x_cuts,y_cuts,z_cuts,numrow,numcol,numplane)
    max_times.append(ss.optimized_tts_3d_numerical(params,im1points,gxmin,gxmax,gymin,gymax,gzmin,gzmax,numrow,numcol,numplane,machine_parameters,num_angles,Am,Az,add_cells,unweighted,test))
    

np.savetxt("max_times_im1_opt",max_times)