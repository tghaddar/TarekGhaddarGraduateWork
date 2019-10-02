##This library contains all mesh related functions.
import numpy as np
import math
from scipy import integrate
from utilities import get_ij
from shapely.geometry import Polygon,Point,MultiPoint,LineString

#Get overlap of line segment
def get_overlap(min1, max1, min2, max2):
    return max(0, min(max1, max2) - max(min1, min2))

#Getting the cells in a subset.
def get_cells_per_subset_2d_numerical(points,boundaries,adjacency_matrix,numrow,numcol):
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
  #Time to adjust the number of cells to take into account cell creation from cuts.
  for s in range(0,num_subsets):
    neighbors = [n for n in range(0,num_subsets) if adjacency_matrix[s][n]==1]
    i_s,j_s = get_ij(s,numrow,numcol)
    #Adding cells where needed based on cut lines.
    for n in neighbors:
      Ly_neighbor = boundaries[n][3] - boundaries[n][2]
      i_n,j_n = get_ij(n,numrow,numcol)
      boundary = 'x'
      if i_s == i_n:
        boundary == 'y'
      
      if boundary == 'y':
        bdy_cells = bdy_cells_per_subset[n][0]
        cells_per_subset[s] += int(bdy_cells/2.0)
      
      if boundary == 'x':
        overlap = get_overlap(boundaries[s][2],boundaries[s][3],boundaries[n][2],boundaries[n][3])
        bdy_cells = bdy_cells_per_subset[n][1]*overlap/Ly_neighbor
        cells_per_subset[s]+= int(bdy_cells/2.0)
    
  return cells_per_subset,bdy_cells_per_subset

def get_cells_per_subset_2d_test(points,boundaries,adjacency_matrix,numrow,numcol,add_cells):
  num_points = len(points[0])
  #Number of subsets.
  num_subsets = len(boundaries)
  #Stores the number of cells per subset.
  cells_per_subset = [0]*num_subsets
  #Stores the number of boundary cells per subset.
  bdy_cells_per_subset = [0.0]*num_subsets  
  xpoints = points[0,:]
  ypoints = points[1,:]

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
    
    #Looping through the points and assigning them to the subset if they fit.
    x1 = np.argwhere(np.logical_and(xpoints >= xmin, xpoints <= xmax)).flatten()
    y1 = ypoints[x1]
    y2 = np.argwhere(np.logical_and(y1 >= ymin, y1 <= ymax)).flatten()
    num_cells = len(y2)
    cells_per_subset[s] = num_cells
    if cells_per_subset[s] == 0:
      cells_per_subset[s] = 1
    
    N = cells_per_subset[s]
    
    #Computing the boundary cells along x and y.
    nx = math.sqrt(N/subset_area)*Lx
    ny =  math.sqrt(N/subset_area)*Ly
    if nx < 1:
      nx = 1
    if ny < 1:
      ny = 1
    bdy_cells_per_subset[s] = [nx,ny]
  
  #Time to adjust the number of cells to take into account cell creation from cuts.
  if (add_cells):
    for s in range(0,num_subsets):
      neighbors = [n for n in range(0,num_subsets) if adjacency_matrix[s][n]==1]
      i_s,j_s = get_ij(s,numrow,numcol)
      #Adding cells where needed based on cut lines.
      for n in neighbors:
        Ly_neighbor = boundaries[n][3] - boundaries[n][2]
        i_n,j_n = get_ij(n,numrow,numcol)
        boundary = 'x'
        if i_s == i_n:
          boundary == 'y'
        
        if boundary == 'y':
          bdy_cells = bdy_cells_per_subset[n][0]
          cells_per_subset[s] += int(np.ceil(bdy_cells/2.0))
        
        if boundary == 'x':
          overlap = get_overlap(boundaries[s][2],boundaries[s][3],boundaries[n][2],boundaries[n][3])
          bdy_cells = bdy_cells_per_subset[n][1]*overlap/Ly_neighbor
          cells_per_subset[s]+= int(np.ceil(bdy_cells/2.0))
      
  return cells_per_subset,bdy_cells_per_subset

    
def which_bounds(intersect,bounds):
  
  int_bounds = []
  for i in range(0,len(bounds)):
    bound= bounds[i]
    if bound.intersects(intersect):
      int_bounds.append(i)
  
  return int_bounds

def tri_area(pt1,pt2,pt3):
  
  area = 0.5*( (pt2[0]-pt1[0])*(pt3[1] - pt1[1]) - (pt3[0] - pt1[0])*(pt2[1]-pt1[1]))
  
  return area

def check_colinear(bound,bound_intersections):
  is_colinear = False
  bound_coords = bound.coords[:]
  int_coords = bound_intersections.coords[:]
  pt1 = bound_coords[0]
  pt2 = bound_coords[1]
  pt3 = int_coords[0]
  area1 = tri_area(pt1,pt2,pt3)
  pt3 = int_coords[1]
  area2 = tri_area(pt1,pt2,pt3)
  
  if area1 == 0.0 and area2 == 0.0:
    is_colinear = True
  
  return is_colinear

def check_nat_boundary(polygon,bound):
  is_nat_boundary = False
  bound_intersections = bound.intersection(polygon)
  intersect_coords = bound_intersections.coords[:]
  num_intersections = len(bound_intersections.coords[:])
  if (num_intersections == 1):
    if intersect_coords[0] in polygon.exterior.coords[:]:
      return True
    elif intersect_coords[0] in bound.coords[:]:
      return True
    else:
      return is_nat_boundary
  elif (num_intersections < 1):
    raise Exception("Less than one intersection. Shouldn't be possible at this point.")
  else:
    ctr  = 0
    for j in range(0,num_intersections):
      if intersect_coords[j] in polygon.exterior.coords[:]:
        ctr += 1
    if ctr == 2:
      is_nat_boundary = True
    elif ctr >= 1:    
      is_colinear = check_colinear(bound,bound_intersections)
      if is_colinear:
        is_nat_boundary = True
    else:
      return is_nat_boundary

  return is_nat_boundary


def check_add_cell(polygon,int_bounds,bounds):
  
  add_cell = [False,False,False,False]
  add_bdy_cell = [False,False,False,False]
  #Checking for a natural boundary for the polygon.
  for i in int_bounds:
    bound = bounds[i]
    add_bdy_cell[i] = True
    #Checking if this boundary intersection is a natural boundary. 
    is_nat_boundary = check_nat_boundary(polygon,bound)
    if is_nat_boundary == False:
      add_cell[i] = True
    
  return add_cell,add_bdy_cell


def get_cells_per_subset_2d_robust(cell_verts,vert_data,boundaries,adjacency_matrix,numrow,numcol):
  #Number of subsets.
  num_subsets = len(boundaries)
  #Stores the number of cells per subset.
  cells_per_subset = [0]*num_subsets
  #Stores the number of boundary cells per boundary per subset.
  bdy_cells_per_subset = [[0,0]]*num_subsets  
  
  numcells = len(cell_verts)
  polygons = [None]*numcells
  
  #Building the cells.
  for i in range(0,numcells):
    current_cell = cell_verts[i]
    current_verts = []
    for p in current_cell:
      #current point
      cp = (vert_data[p][0],vert_data[p][1])
      current_verts.append(cp)
    
    polygons[i] = MultiPoint(current_verts).convex_hull
  
  #Looping through the subsets.
  for s in range(0,num_subsets):
    
    subset_bounds = boundaries[s]
    #neighbors = [n for n in range(0,num_subsets) if adjacency_matrix[s][n]==1]
    xmin = subset_bounds[0]
    xmax = subset_bounds[1]
    ymin = subset_bounds[2]
    ymax = subset_bounds[3]
    
    subset = MultiPoint([(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)]).convex_hull

    for polygon in polygons:
      if polygon.within(subset):
        cells_per_subset[s] += 1
        #polygons.remove(polygon)
      elif polygon.intersects(subset):
        #Checking for natural boundaries. If the only intersection is a natural boundary on the exterior of the subset, then we don't add the cell to this subset.
        if polygon.touches(subset):
          continue
        else:
          cells_per_subset[s] += 1
    
    #The x length of the subset.
    Lx = xmax - xmin
    #The y length of the subset.
    Ly = ymax - ymin
    #The area of the subset.
    subset_area = Lx*Ly
    N = cells_per_subset[s]
    #Computing the boundary cells along x and y.
    nx = math.sqrt(N/subset_area)*Lx
    ny =  math.sqrt(N/subset_area)*Ly
    if nx < 1:
      nx = 1
    if ny < 1:
      ny = 1
    bdy_cells_per_subset[s] = [nx,ny]
    
  return cells_per_subset,bdy_cells_per_subset

def get_cells_per_subset_3d_numerical(points,boundaries):
  #Number of points in the domain.
  num_points = len(points[0])
  #The total number of subsets.
  num_subsets = len(boundaries)  
  #Stores the number of cells per subset.
  cells_per_subset = [0]*num_subsets
  #Stores the number of boundary cells per subset.
  bdy_cells_per_subset = [0.0]*num_subsets
  
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
    
    #Looping through the points and assigning them to the subset if they fit.
    for p in range(0,num_points):
      xpoint = points[0][p]
      ypoint = points[1][p]
      zpoint = points[2][p]
      
      if zpoint >= zmin and zpoint <= zmax:
        if xpoint >= xmin and xpoint <= xmax:
          if ypoint >= ymin and ypoint <= ymax:
            cells_per_subset[s] += 1
      
    #If there are no centroids in the subset, then the subset boundaries form a cell.
    if cells_per_subset[s] == 0:
      cells_per_subset[s] = 1
      
    N = cells_per_subset[s]
    #Computing boundary cells along xy, xz, and yz faces.
    n_xy = pow((N/subset_vol),2.0/3.0)*Lx*Ly
    n_xz = pow((N/subset_vol),2.0/3.0)*Lx*Lz
    n_yz = pow((N/subset_vol),2.0/3.0)*Ly*Lz
    bdy_cells_per_subset[s] = [n_xy,n_xz,n_yz]
  
  return cells_per_subset,bdy_cells_per_subset
  
def get_cells_per_subset_3d_numerical_test2(points,boundaries,add_cells):
  #Number of points in the domain.
  num_points = len(points[0])
  #The total number of subsets.
  num_subsets = len(boundaries)  
  #Stores the number of cells per subset.
  cells_per_subset = [0]*num_subsets
  #Stores the number of boundary cells per subset.
  bdy_cells_per_subset = [0.0]*num_subsets
  xpoints = points[0,:]
  ypoints = points[1,:]
  zpoints = points[2,:]
  
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
    
    #Looping through the points and assigning them to the subset if they fit.
    x1 = np.argwhere(np.logical_and(xpoints >= xmin, xpoints <= xmax)).flatten()
    y1 = ypoints[x1]
    z1 = zpoints[x1]
    y2 = np.argwhere(np.logical_and(y1 >= ymin, y1 <= ymax)).flatten()
    z2 = z1[y2]
    z3 = np.argwhere(np.logical_and(z2 >= zmin, z2 <= zmax)).flatten()
    num_cells = len(z3)
    cells_per_subset[s] = num_cells
   
      
    #If there are no centroids in the subset, then the subset boundaries form a cell.
    if cells_per_subset[s] == 0:
      cells_per_subset[s] = 1
      
    N = cells_per_subset[s]
    #Computing boundary cells along xy, xz, and yz faces.
    n_xy = pow((N/subset_vol),2.0/3.0)*Lx*Ly
    n_xz = pow((N/subset_vol),2.0/3.0)*Lx*Lz
    n_yz = pow((N/subset_vol),2.0/3.0)*Ly*Lz
    bdy_cells_per_subset[s] = [n_xy,n_xz,n_yz]
  
  return cells_per_subset,bdy_cells_per_subset
def get_cells_per_subset_3d_numerical_test(points,boundaries):
  #Number of points in the domain.
  num_points = len(points[0])
  #The total number of subsets.
  num_subsets = len(boundaries)  
  #Stores the number of cells per subset.
  cells_per_subset = [0]*num_subsets
  #Stores the number of boundary cells per subset.
  bdy_cells_per_subset = [0.0]*num_subsets
  
  #Looping through the points and assigning them to the subset if they fit.
  for p in range(0,num_points):
    xpoint = points[0][p]
    ypoint = points[1][p]
    zpoint = points[2][p]

    for s in range(0,num_subsets):
      #The boundaries of this subset.
      subset_bounds = boundaries[s]
      xmin = subset_bounds[0]
      xmax = subset_bounds[1]
      ymin = subset_bounds[2]
      ymax = subset_bounds[3]
      zmin = subset_bounds[4]
      zmax = subset_bounds[5]
    
      if zpoint >= zmin and zpoint <= zmax:
        if xpoint >= xmin and xpoint <= xmax:
          if ypoint >= ymin and ypoint <= ymax:
            cells_per_subset[s] += 1
            break
    
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
    #If there are no centroids in the subset, then the subset boundaries form a cell.
    if cells_per_subset[s] == 0:
      cells_per_subset[s] = 1
      
    N = cells_per_subset[s]
    #Computing boundary cells along xy, xz, and yz faces.
    n_xy = pow((N/subset_vol),2.0/3.0)*Lx*Ly
    n_xz = pow((N/subset_vol),2.0/3.0)*Lx*Lz
    n_yz = pow((N/subset_vol),2.0/3.0)*Ly*Lz
    bdy_cells_per_subset[s] = [n_xy,n_xz,n_yz]
  
  return cells_per_subset,bdy_cells_per_subset
#Integrating an analytical mesh density function.
#f = the analytical description of the mesh density function.
def analytical_mesh_integration_3d(f,xmin,xmax,ymin,ymax,zmin,zmax):
  
  return integrate.tplquad(f,xmin,xmax,lambda x: ymin, lambda x: ymax, lambda x,y: zmin, lambda x,y: zmax)

def get_z_cells(f,zmin,zmax):
  return integrate.tplquad(f,zmin,zmax)

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
    
    N = analytical_mesh_integration_3d(f,xmin,xmax,ymin,ymax,zmin,zmax)[0]
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
  mid_range = range(0,nx)
  y_cuts_j = [y_cuts_i for i in mid_range]
  final_range = range(0,nz)
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
