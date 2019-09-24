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
from copy import copy
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
  
  #The discrete x_steps we are using to build the cdf. Equivalent to 1% of column width if using even cuts.
  num_steps = int((gxmax-gxmin)/(0.01*(gxmax - gxmin)/numcol))
  #The number of bins in the CDF.
  hist_range = (gxmin,gxmax)
  #Building a histogram
  hist,bin_edges = np.histogram(points,bins=num_steps,range=hist_range,normed=False)
  
  cdf = np.cumsum(hist)
  cdf = cdf/max(cdf)
  cdf = np.insert(cdf,0,0.0)
  
  return cdf,bin_edges
  
def get_row_cdf(points,gymin,gymax,numrow):
  
  num_steps = int((gymax-gymin)/(0.001*(gymax - gymin)/numrow))
   #The number of bins in the CDF.
  hist_range = (gymin,gymax)
  #Building a histogram
  hist,bin_edges = np.histogram(points,bins=num_steps,range=hist_range,normed=False)
  
  cdf = np.cumsum(hist)
  cdf = cdf/max(cdf)
  cdf = np.insert(cdf,0,0.0)
  
  return cdf,bin_edges

def get_highest_jumps(points,gmin,gmax,numdim):
  
  #The discrete steps we are using to build the cdf. Equivalent to 1% of column width if using even cuts.
  num_steps = int((gmax-gmin)/(0.001*(gmax - gmin)/numdim))
  #The number of bins in the CDF.
  hist_range = (gmin,gmax)
  #Building a histogram
  hist,bin_edges = np.histogram(points,bins=num_steps,range=hist_range,normed=False)
  
  cdf = np.cumsum(hist)
  cdf = cdf/max(cdf)
  cdf = np.insert(cdf,0,0.0)
  
  #Getting the derivate to identify the highest jumps.
  grad_cdf = np.diff(cdf)/np.diff(bin_edges)
  highest_jumps = np.argsort(grad_cdf)[-(numdim-1):]
  values = bin_edges[highest_jumps]
  values = np.sort(values)
  values = np.append(values,gmax)
  values = np.insert(values,0,gmin)
  
  return values

def create_opt_cut_suite(points,gxmin,gxmax,gymin,gymax,numcol,numrow):
  
  #The columnar cdfs.
  column_cdf,column_bin_edges = get_column_cdf(points,gxmin,gxmax,numcol)
  #Getting the gradient to identify the jumps.
  grad_cdf = np.diff(column_cdf)/np.diff(column_bin_edges)
  #Normalizing.
  norm = grad_cdf/max(grad_cdf)
  #Getting the highest jumps.
  highest_jumps = np.argsort(norm)[-(numcol-1):]
  #The x_values corresponding to the highest jumps.
  x_values = column_bin_edges[highest_jumps]
  x_values = np.sort(x_values)
  x_values = np.append(x_values,gxmax)
  x_values = np.insert(x_values,0,gxmin)
  
  #Looping over columns to get the row-wise cdfs in each column.
  xpoints = points[:,0]
  ypoints = points[:,1]
  all_y_cuts = []
  for col in range(1,numcol+1):
    xmin = x_values[col-1]
    xmax = x_values[col]
    x1 = np.argwhere(np.logical_and(xpoints>=xmin,xpoints<=xmax)).flatten()
    #Pulling all points that are in this column. 
    y1 = ypoints[x1]
    #Getting the highest jumps for this column.
    y_values_col = get_highest_jumps(y1,gymin,gymax,numrow)
#    row_cdf, row_bin_edges = get_row_cdf(y1,gymin,gymax,numrow)
#    grad_row_cdf = np.diff(row_cdf)/np.diff(row_bin_edges)
#    norm_row = grad_row_cdf/max(grad_row_cdf)
#    highest_row_jumps = np.argsort(norm_row)[-(numrow-1):]
#    y_values_col = row_bin_edges[highest_row_jumps]
#    y_values_col = np.sort(y_values_col)
#    y_values_col = np.append(y_values_col,gymax)
#    y_values_col = np.insert(y_values_col,0,gymin)
    all_y_cuts.append(y_values_col)
    
  #Doing a binary tree of the columns to get a full cut suite.
  tree_bottom = False
  num_children = 2
  prev_x_limits = [numcol]
  y_cut_suite = []
  
  #Getting all the way through cuts.
  y_values_all_through = get_highest_jumps(ypoints,gymin,gymax,numrow)
  all_through_vals = []
  for i in range(0,numcol):
    all_through_vals.append(y_values_all_through)
    
  y_cut_suite.append(all_through_vals)
  
  while(tree_bottom == False):
    
    x_limits = []
    current_x_limit = int(0)
    current_y_values = [[] for i in range(0,numcol)]
    for i in range(0,len(prev_x_limits)):
      x1_limit = int(np.floor(prev_x_limits[i]/2))
      x_limits.append(x1_limit)
      x2_limit = int(np.ceil(prev_x_limits[i]/2))
      x_limits.append(x2_limit)
      
      col0 = copy(current_x_limit)
      xmin = x_values[col0]
      current_x_limit += x1_limit
      col1 = copy(current_x_limit)
      xmax = x_values[col1]
      
      xverts0 = np.argwhere(np.logical_and(xpoints>=xmin,xpoints<=xmax)).flatten()
      y0 = ypoints[xverts0]
      y_values0 = get_highest_jumps(y0,gymin,gymax,numrow)
      for j in range(col0,col1):
        current_y_values[j] = y_values0
      #current_y_values[col0:col1] = [y_values0]
      
      xmin2 = xmax
      current_x_limit += x2_limit
      col2 = copy(current_x_limit)
      xmax2 = x_values[col2]
      
      xverts1 = np.argwhere(np.logical_and(xpoints>=xmin2,xpoints<=xmax2)).flatten()
      y1 = ypoints[xverts1]
      y_values1 = get_highest_jumps(y1,gymin,gymax,numrow)
      for j in range(col1,col2):
        current_y_values[j] = y_values1
      #current_y_values[col1:col2] = [y_values1]
    
    y_cut_suite.append(current_y_values)  
    #print(current_y_values)
    print(x_limits)
      
#    for i in range(0,2):
#      prev_x_limit = current_x_limit
#      if i%2 == 0:
#        current_x_limit += x1_limit
#      else:
#        current_x_limit += x2_limit
#      print(num_children,current_x_limit)
#      xmin = x_values[prev_x_limit]
#      xmax = x_values[current_x_limit]
      
    num_children *= 2
    prev_x_limits = x_limits
    if (x_limits[0] <= 1.5):
      tree_bottom = True
      
  y_cut_suite.append(all_y_cuts)
  return x_values,y_cut_suite