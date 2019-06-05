from sweep_solver import unpack_parameters_3d
from optimizer import create_parameter_space_3d,create_bounds_3d
from mesh_processor import create_3d_cuts

#Global boundaries.
global_x_min = 0.0
global_x_max = 10.0
global_y_min = 0.0
global_y_max = 10.0
global_z_min = 0.0
global_z_max = 10.0

numplane = 2
numcol = 2
numrow = 2


z_cuts_og,x_cuts_og,y_cuts_og = create_3d_cuts(global_x_min,global_x_max,numcol,global_y_min,global_y_max,numrow,global_z_min,global_z_max,numplane)

params = create_parameter_space_3d(x_cuts_og,y_cuts_og,z_cuts_og,numrow,numcol,numplane)
num_params = len(params)


x_cuts,y_cuts,z_cuts= unpack_parameters_3d(params,global_x_min,global_x_max,global_y_min,global_y_max,global_z_min,global_z_max,numcol,numrow,numplane)

bounds = create_bounds_3d(num_params,global_x_min,global_x_max,global_y_min,global_y_max,global_z_min,global_z_max,numrow,numcol,numplane)

if (z_cuts_og == z_cuts):
  print("Z checks out")
else:
  print("Z fail")
  
if (x_cuts_og == x_cuts):
  print("X checks out")
else:
  print("X fail")
  
if (y_cuts_og == y_cuts):
  print("Y checks out")
else:
  print("Y fail")