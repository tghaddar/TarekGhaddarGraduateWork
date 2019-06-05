#This library contains all mesh related functions.
import numpy as np
import math
from scipy import integrate

#Getting the cells in a subset.
def get_cells_per_subset_2d_numerical(points,boundaries):
  num_points = len(points[0])
  #Number of subsets.
  num_subsets = len(boundaries)
  #Stores the number of cells per subset.
  cells_per_subset = [0]*num_subsets
  #Stores the number of boundary cells per subset.
  bdy_cells_per_subset = [0.0]*num_subsets  
  
  #Looping through the subsets.
  for s in range(0,num_subsets):
    
    subset_bounds = boundaries[s]
    xmin = subset_bounds[0]
    xmax = subset_bounds[1]
    ymin = subset_bounds[2]
    ymax = subset_bounds[3]
    
    #The x length of the subset.
    Lx = xmax - xmin
    #The y length of the subset.
    Ly = ymax - ymin
    #The area of the subset.
    subset_area = Lx*Ly
    
    for p in range(0,num_points):
      xpoint = points[0][p]
      ypoint = points[1][p]
      
      if xpoint >= xmin and xpoint <= xmax:
        if ypoint >= ymin and ypoint <=ymax:
          cells_per_subset[s] += 1
    if cells_per_subset[s] == 0:
      cells_per_subset[s] = 1
      
    N = cells_per_subset[s]
    #Computing the boundary cells along x and y.
    nx = math.sqrt(N/subset_area)*Lx
    ny =  math.sqrt(N/subset_area)*Ly
    bdy_cells_per_subset[s] = [nx,ny]
    
  return cells_per_subset,bdy_cells_per_subset
  
#Integrating an analytical mesh density function.
#f = the analytical description of the mesh density function.
def analytical_mesh_integration_3d(f,xmin,xmax,ymin,ymax,zmin,zmax):
  
  return integrate.tplquad(f,xmin,xmax,lambda x: ymin, lambda x: ymax, lambda x,y: zmin, lambda x,y: zmax)

def analytical_mesh_integration_2d(f,xmin,xmax,ymin,ymax):
  return integrate.dblquad(f,xmin,xmax,lambda x: ymin, lambda x: ymax)

def get_cells_per_subset_3d(f,boundaries):
  #The total number of subsets.
  num_subsets = len(boundaries)  
  #Stores the number of cells per subset.
  cells_per_subset = [None]*num_subsets
  #Stores the number of boundary cells per subset.
  bdy_cells_per_subset = [None]*num_subsets
  
  #Looping through the subsets.
  for s in range(0,num_subsets):
    
    #The boundaries of this subset.
    subset_bounds = boundaries[s]
    xmin = subset_bounds[0]
    xmax = subset_bounds[1]
    ymin = subset_bounds[2]
    ymax = subset_bounds[3]
    zmin = subset_bounds[4]
    zmax = subset_bounds[5]
    
    #The x,y, and z lengths of the subset.
    Lx = xmax - xmin
    Ly = ymax - ymin
    Lz = zmax - zmin
    #Subset volume.
    subset_vol = Lx*Ly*Lz
    
    N = analytical_mesh_integration_3d(f,xmin,xmax,ymin,ymax,zmin,zmax)
    cells_per_subset[s] = N
    
    #Computing boundary cells along xy, xz, and yz faces.
    n_xy = pow((N/subset_vol),2.0/3.0)*Lx*Ly
    n_xz = pow((N/subset_vol),2.0/3.0)*Lx*Lz
    n_yz = pow((N/subset_vol),2.0/3.0)*Ly*Lz
    bdy_cells_per_subset[s] = [n_xy,n_xz,n_yz]
  
  return cells_per_subset,bdy_cells_per_subset

#Getting the cells and boundary cells per subset.
def get_cells_per_subset_2d(f,boundaries):
  
  #The total number of subsets.
  num_subsets = len(boundaries)  
  #Stores the number of cells per subset.
  cells_per_subset = [None]*num_subsets
  #Stores the number of boundary cells per subset.
  bdy_cells_per_subset = [None]*num_subsets
  
  #Looping through the subsets.
  for s in range(0,num_subsets):
    
    subset_bounds = boundaries[s]
    xmin = subset_bounds[0]
    xmax = subset_bounds[1]
    ymin = subset_bounds[2]
    ymax = subset_bounds[3]
    
    #The x length of the subset.
    Lx = xmax - xmin
    #The y length of the subset.
    Ly = ymax - ymin
    #The area of the subset.
    subset_area = Lx*Ly
    
    #Getting the number of cells in the current subset.
    N = analytical_mesh_integration_2d(f,xmin,xmax,ymin,ymax)[0]
    cells_per_subset[s] = N
    
    #Computing the boundary cells along x and y.
    nx = math.sqrt(N/subset_area)*Lx
    ny =  math.sqrt(N/subset_area)*Ly
    bdy_cells_per_subset[s] = [nx,ny]
    

  return cells_per_subset,bdy_cells_per_subset
    

#Creates uniform 3d cuts given boundaries and number of subsets in each dimension.
def create_3d_cuts(xmin,xmax,nx,ymin,ymax,ny,zmin,zmax,nz):
  
  #The z_cuts.
  zstep = (zmax- zmin)/nz
  z_range = range(0,nz+1)
  z_cuts = [zmin+i*zstep for i in z_range]
  
  #The x_cuts.
  xstep = (xmax - xmin)/nx
  x_range = range(0,nx+1)
  x_cuts_i = [xmin+i*xstep for i in x_range]
  final_range = range(0,nz)
  x_cuts = [x_cuts_i for i in final_range]
  
  #The y_cuts.
  ystep = (ymax - ymin)/ny
  y_range = range(0,ny+1)
  y_cuts_i = [ymin + i*ystep for i in y_range]
  mid_range = range(0,nz)
  y_cuts_j = [y_cuts_i for i in mid_range]
  final_range = range(0,nx)
  y_cuts = [y_cuts_j for i in final_range]
  
  
  return z_cuts,x_cuts,y_cuts

def create_2d_cuts(xmin,xmax,nx,ymin,ymax,ny):
  
  #The x_cuts.
  xstep = (xmax- xmin)/nx
  x_range = range(0,nx+1)
  x_cuts = [xmin+i*xstep for i in x_range]
  
  #The y cuts.
  ystep = (ymax - ymin)/ny
  y_range = range(0,ny+1)
  y_cuts_i = [ymin+i*ystep for i in y_range]
  final_range = range(0,nx)
  y_cuts = [y_cuts_i for i in final_range]
  
  return x_cuts,y_cuts
  

def create_2d_cut_suite(xmin,xmax,nx,ymin,ymax,ny):
  
  num_steps = 100
  
  xstep = (xmax - xmin)/num_steps
  ystep = (ymax - ymin)/num_steps
  
  all_x_cuts = [None]*(num_steps-1)
  all_y_cuts = [None]*pow(num_steps-1,2)
  
  for i in range(0,num_steps-1):
    x_cuts = [xmin, xmin+(i+1)*xstep, xmax]
    all_x_cuts[i] = x_cuts
  
  counter = 0
  for i in range(0,num_steps-1):
    for j in range(0,num_steps-1):
      
      y_cuts_i= [ymin, ymin+(i+1)*ystep,ymax]
      y_cuts_j = [ymin,ymin+(j+1)*ystep,ymax]
      all_y_cuts[counter] = [y_cuts_i,y_cuts_j]
      counter += 1
      
      
  return all_x_cuts,all_y_cuts
