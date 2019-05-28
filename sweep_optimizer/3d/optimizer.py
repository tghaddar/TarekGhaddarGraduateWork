#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 14:52:23 2018
This is the driver file for the sweep optimizer.
@author: tghaddar
"""
from mesh_processor import analytical_mesh_integration_2d,create_2d_cuts,get_cells_per_subset_2d,create_2d_cut_suite
from sweep_solver import unpack_parameters,optimized_tts
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
from scipy.optimize import minimize
warnings.filterwarnings("ignore")

plt.close("all")

def create_parameter_space(x_cuts,y_cuts,numrow,numcol):

  interior_cuts = [x_cuts[i] for i in range(1,numcol)]

  for col in range(0,numcol):
    interior_y_cuts = [y_cuts[col][i] for i in range(1,numrow)]
    interior_cuts += interior_y_cuts
    
  return interior_cuts

def create_bounds(num_params,global_xmin,global_xmax,global_ymin,global_ymax,numrow,numcol):
  
  x_tol = 0.05*(global_xmax - global_xmin)/numcol
  y_tol = 0.05*(global_ymax - global_ymin)/numrow
  nx = numcol - 1
  cut_id = 0
  bounds = [()for i in range(0,num_params)]
  for i in range(0,num_params):
    if cut_id < nx:
      bounds[i] += (global_xmin+x_tol,global_xmax-x_tol)
    else:
      bounds[i] += (global_ymin+y_tol,global_ymax-y_tol)
    
    cut_id += 1
  return bounds

#The machine parameters.
#Communication time per double
#t_comm = 4.47e-02
##The number of bytes to communicate per subset.
##The message latency time.
#m_l = 1
#latency = 4110.0e-02
##Solve time per unknown.
#t_u = 450.0e-02
#upc = 4.0
#upbc = 2.0
#
#machine_params = (t_u,upc,upbc,t_comm,latency,m_l)
#
##Number of rows and columns.
#numrow = 3
#numcol = 3
#
##Global boundaries.
#global_xmin = 0.0
#global_xmax = 10.0
#global_ymin = 0.0
#global_ymax = 10.0
#
##The subset boundaries.
#x_cuts,y_cuts = create_2d_cuts(global_xmin,global_xmax,numcol,global_ymin,global_ymax,numrow)
#
#interior_cuts = create_parameter_space(x_cuts,y_cuts,numrow,numcol)
#num_params = len(interior_cuts)
#bounds = create_bounds(num_params,global_xmin,global_xmax,global_ymin,global_ymax,numcol)
#
#test_x_cuts,test_y_cuts = unpack_parameters(interior_cuts,global_xmin,global_xmax,global_ymin,global_ymax,numcol,numrow)
#
##The mesh density function.
#f = lambda x,y: x + y
#
#args = (f,global_xmin,global_xmax,global_ymin,global_ymax,numrow,numcol,t_u,upc,upbc,t_comm,latency,m_l)
#
#max_time = minimize(optimized_tts,interior_cuts,args = args,bounds = bounds,options={'maxiter':10,'maxfun':10,'disp':False})



