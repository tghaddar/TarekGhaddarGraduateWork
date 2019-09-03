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
import numpy as np
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
    interior_x_cuts = [x_cuts[plane][i] for i in range(1,numcol)]
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

def create_constraints(global_xmin,global_xmax,global_ymin,global_ymax,numrow,numcol):
  
  x_tol = (0.05/numcol)*(global_xmax - global_xmin)/numcol
  y_tol = (0.05/numrow)*(global_ymax - global_ymin)/numrow
  
  #The number of constraints for each dimension.
  num_cons_x = numcol-2
  num_cons_y = numcol*(numrow-2)
  #LIst of dictionaries storing the constraints.
  constraints = [None]*(num_cons_x+num_cons_y)
  print(len(constraints))

  for xcons in range(0,num_cons_x):
    current_constraint = {}
    current_constraint['type'] = 'ineq'
    current_constraint['fun'] = lambda x: x[xcons+1] - x[xcons] - x_tol
    constraints[xcons] = current_constraint

  for ycons in range(num_cons_x,num_cons_x+num_cons_y):
    current_constraint = {}
    current_constraint['type'] = 'ineq'
    current_constraint['fun'] = lambda x: x[ycons+1] - x[ycons] - y_tol
    constraints[ycons] = current_constraint 

  return constraints
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

def get_column_cdf(points,gxmin,gxmax,numcol):
  
  #x coordinates of the centroid distribution.
  x_points = points[:,0]
  #The discrete x_steps we are using to build the cdf. Equivalent to 1% of column width if using even cuts.
  num_steps = int((gxmax-gxmin)/(0.01*(gxmax - gxmin)/numcol))
  #The number of bins in the CDF.
  hist_range = (gxmin,gxmax)
  #Building a histogram
  hist,bin_edges = np.histogram(x_points,bins=num_steps,range=hist_range,normed=False)
  
  cdf = np.cumsum(hist)
  cdf = cdf/max(cdf)
  cdf = np.insert(cdf,0,0.0)
  
  return cdf,bin_edges
  
def get_row_cdf(points,gymin,gymax,numrow):
  y_points = points[:,1]
  num_steps = int((gymax-gymin)/(0.01*(gymax - gymin)/numrow))
   #The number of bins in the CDF.
  hist_range = (gymin,gymax)
  #Building a histogram
  hist,bin_edges = np.histogram(y_points,bins=num_steps,range=hist_range,normed=False)
  
  cdf = np.cumsum(hist)
  cdf = cdf/max(cdf)
  
  return cdf,bin_edges
