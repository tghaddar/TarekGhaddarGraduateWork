#This library contains all mesh related functions.
import numpy as np
from scipy import integrate


#Integrating an analytical mesh density function.
#f = the analytical description of the mesh density function.
def analytical_mesh_integration(f,xmin,xmax,ymin,ymax,zmin,zmax):
  
  return integrate.tplquad(f,xmin,xmax,lambda x: ymin, lambda x: ymax, lambda x,y: zmin, lambda x,y: zmax)
  

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