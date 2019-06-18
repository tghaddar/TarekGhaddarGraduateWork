import numpy as np
import sys
sys.path.append(r'C:\Users\tghad\Documents\GitHub\TarekGhaddarGraduateWork\sweep_optimizer\3d')
import build_3d_adjacency as b3a
import sweep_solver as ss
from mesh_processor import create_3d_cuts,get_cells_per_subset_3d_numerical_test
from utilities  import get_ijk
from optimizer import create_parameter_space_3d

#Communication time per double
t_comm = 4.47e-09
#The number of bytes to communicate per subset.
#The message latency time.
m_l = 1
latency = 4110.0e-09
#Solve time per unknown.
t_u = 3500.0e-09
upc = 4.0
upbc = 2.0

gxmin = 0.0
gxmax = 60.96
gymin = 0.0
gymax = 60.96
gzmin = 0.0
gzmax = 146.05

num_angles = 1
unweighted = True
test = False
num_row = 5
num_col = 5
num_plane = 5
subsets=125

min_max = [[None]*6 for i in range(0,subsets)]
z_cuts = [None]*(num_plane+1)
z_cuts[0] = gzmin
x_cuts = [[gxmin]*6 for i in range(0,num_plane)]
y_cuts = [[gymin]*6 for i in range(0,num_col)]
y_cuts = [y_cuts for i in range(0,num_plane)]
for i in range(0,subsets):
    
  filename = "points"+str(i)
  
  f = open(filename,'r')
  points = f.readlines()
  points = points[18:]
  points = [x.strip() for x in points]
  num_points = int(points[0])
  points.pop(0)
  points.pop(0)
  for p in range(0,num_points):
    line = points[p]
    line = line[1:]
    line = line[:-1]
    line = line.split()
    line = [float(x) for x in line]
    points[p] = line
  
  points = np.array(points[:num_points])
  xmin = np.min(points[:,0])
  xmax = np.max(points[:,0])
  ymin = np.min(points[:,1])
  ymax = np.max(points[:,1])
  zmin = np.min(points[:,2])
  zmax = np.max(points[:,2])
  min_max[i] = [xmin,xmax,ymin,ymax,zmin,zmax]
  f.close()
  
  i,j,k = get_ijk(i,num_row,num_col,num_plane)
  z_cuts[k+1] = zmax
  x_cuts[k][i+1] = xmax
  y_cuts[k][i][j+1] = ymax
  

im1points = np.genfromtxt("../openfoam_mesh/im1_cell_centers")
params = create_parameter_space_3d(x_cuts,y_cuts,z_cuts,num_row,num_col,num_plane)
max_time_lbd = ss.optimized_tts_3d_numerical(params,im1points,gxmin,gxmax,gymin,gymax,gzmin,gzmax,num_row,num_col,num_plane,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted,test)

z_cuts,x_cuts,y_cuts = create_3d_cuts(gxmin,gxmax,num_col,gymin,gymax,num_row,gzmin,gzmax,num_plane)
params = create_parameter_space_3d(x_cuts,y_cuts,z_cuts,num_row,num_col,num_plane)
max_time_reg = ss.optimized_tts_3d_numerical(params,im1points,gxmin,gxmax,gymin,gymax,gzmin,gzmax,num_row,num_col,num_plane,t_u,upc,upbc,t_comm,latency,m_l,num_angles,unweighted,test)



