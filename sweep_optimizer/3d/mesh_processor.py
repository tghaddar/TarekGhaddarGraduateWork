#This library contains all mesh related functions.
import numpy as np
from scipy import integrate


#Integrating an analytical mesh density function.
#f = the analytical description of the mesh density function.
def analytical_mesh_integration(f,xmin,xmax,ymin,ymax,zmin,zmax):
  
  return integrate.tplquad(f,xmin,xmax,lambda x: ymin, lambda x: ymax, lambda x,y: zmin, lambda x,y: zmax)
  

