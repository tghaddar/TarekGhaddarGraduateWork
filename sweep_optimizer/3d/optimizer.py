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

def create_parameter_space_3d(x_cuts,y_cuts,z_cuts,numrow,numcol,numplane):
  
  interior_cuts = [z_cuts[i] for i in range(1,numplane)]
  
  for plane in range(0,numplane):
    interior_x_cuts = [x_cuts[plane][i] for i in range(1,numplane)]
    interior_cuts += interior_x_cuts
    
  for plane in range(0,numplane):
    for col in range(0,numcol):
      interior_y_cuts = [y_cuts[plane][col][i] for i in range(1,numrow)]
      interior_cuts += interior_y_cuts
  
  return interior_cuts
  
def create_bounds(num_params,global_xmin,global_xmax,global_ymin,global_ymax,numrow,numcol):
  
  x_tol = (0.05/numcol)*(global_xmax - global_xmin)/numcol
  y_tol = (0.05/numrow)*(global_ymax - global_ymin)/numrow
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

def create_bounds_3d(num_params,global_xmin,global_xmax,global_ymin,global_ymax,global_zmin,global_zmax,numrow,numcol,numplane):
  
  x_tol = (0.05/numcol)*(global_xmax - global_xmin)/numcol
  y_tol = (0.05/numrow)*(global_ymax - global_ymin)/numrow
  z_tol = (0.05/numplane)*(global_zmax - global_zmin)/numplane
  
  nx = numcol-1
  nz = numplane - 1
  
  cut_id = 0
  bounds = [() for i in range(0,num_params)]
  
  for i in range(0,num_params):
    if cut_id < nz:
      bounds[i] += (global_zmin+z_tol,global_zmax-z_tol)
    elif cut_id < nz+numplane*nx:
      bounds[i] += (global_xmin+x_tol,global_xmax-x_tol)
    else:
      bounds[i] += (global_ymin+y_tol,global_ymax-y_tol)
  
  return bounds


