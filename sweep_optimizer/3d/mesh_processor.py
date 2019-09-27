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

    
def check_add_cell(bound,polygon,current_verts):
  
  #Do we add a boundary cell?
  add_boundary_cell = False
  #Do we add a cell to our neighbor?
  add_cell = False
  

  #Checking if the boundary and polygon intersect at all. If so, we add a boundary cell.
  if bound.intersects(polygon):
    add_boundary_cell = True
    #Checking if the cell is on a natural boundary.
    is_nat_boundary = False
    #If it does, we get the intersection points.
    intersect = bound.intersection(polygon)
    intersect_coords = intersect.coords[:]
    num_intersections = len(intersect_coords)
    #If there is more than one point that intersects, we may need to add a cell.
    if (num_intersections > 1):
      #With more than one intersection, we either have a natural boundary (we don't add a cell), or we are slicing through a cell (we add a cell to our neighbor). 
      ctr = 0
      for i in range(0,num_intersections):
        #Check if the intersection is a vertex of the polygon.
        if intersect_coords[i] in current_verts:
          ctr += 1
      #If both intersection points are in the polygon, then it is a natural boundary.
      if ctr == 2:
        is_nat_boundary = True
      if ctr == 0:
        add_cell = True
    #If the cell isn't on a natural boundary, we add the cell to our neighbor.
    if is_nat_boundary == False:
      add_cell = True

    
  return add_cell,add_boundary_cell

  
def get_cells_per_subset_2d_robust(points,cell_verts,vert_data,boundaries,adjacency_matrix,numrow,numcol,add_cells):
  #Number of subsets.
  num_subsets = len(boundaries)
  #Stores the number of cells per subset.
  cells_per_subset = [0]*num_subsets
  #Stores the number of boundary cells per boundary per subset.
  bdy_cells_per_subset = [[0,0,0,0]]*num_subsets  
  xpoints = points[0,:]
  ypoints = points[1,:]

  #Looping through the subsets.
  for s in range(0,num_subsets):
    
    subset_bounds = boundaries[s]
    neighbors = [n for n in range(0,num_subsets) if adjacency_matrix[s][n]==1]
    xmin = subset_bounds[0]
    xmax = subset_bounds[1]
    ymin = subset_bounds[2]
    ymax = subset_bounds[3]
    
    xmin_bound = LineString([(xmin,ymin), (xmin,ymax)])
    xmax_bound = LineString([(xmax,ymin), (xmax,ymax)])
    ymin_bound = LineString([(xmin,ymin), (xmax,ymin)])
    ymax_bound = LineString([(xmin,ymax), (xmax,ymax)])
    
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
    current_cells = [cell_verts[x] for x in y2]
    num_cells = len(y2)
    cells_per_subset[s] = num_cells
    if cells_per_subset[s] == 0:
      cells_per_subset[s] = 1
      bdy_cells_per_subset[s] = [1,1,1,1]
    
    if add_cells:
      if cells_per_subset[s] > 1:
        for cell in range(0,num_cells):
          current_cell = current_cells[cell]
          #numpts = len(current_cell)
          current_verts = []
          for p in current_cell:
            #current point
            cp = (vert_data[p][0],vert_data[p][1])
            current_verts.append(cp)
          
          polygon = MultiPoint(current_verts).convex_hull
          #We check if this cell intersects a bound, and if so, if we need to add the cell to our neighbor.
          if (cell == 8):
            print("debug stop")
          #Checking the xmin bound.
          add_cell,add_boundary_cell = check_add_cell(xmin_bound,polygon,current_verts)
          if add_boundary_cell:
            bdy_cells_per_subset[s][0] += 1
          if add_cell:
            #Get the xmin neighbors of the subset.
            xmin_neighbors = [n for n in neighbors if get_ij(n,numrow,numcol)[0] < get_ij(s,numrow,numcol)[0]]
            #Approximating that we add a cell to every neighbor of the boundary.
            for n in xmin_neighbors:
              cells_per_subset[n] += 1
          
          #Check the xmax bound.
          add_cell,add_boundary_cell = check_add_cell(xmax_bound,polygon,current_verts)
          if add_boundary_cell:
            bdy_cells_per_subset[s][1] += 1
          if add_cell:
            #Get the xmax neighbors of the subset.
            xmax_neighbors = [n for n in neighbors if get_ij(n,numrow,numcol)[0] > get_ij(s,numrow,numcol)[0]]
            #Approximating that we add a cell to every neighbor of the boundary.
            for n in xmax_neighbors:
              cells_per_subset[n] += 1
          
          #Checking the ymin bound.
          add_cell,add_boundary_cell = check_add_cell(ymin_bound,polygon,current_verts)
          if add_boundary_cell:
            bdy_cells_per_subset[s][2] += 1
          if add_cell:
            #Get the ymin neighbors of the subset.
            ymin_neighbors = [n for n in neighbors if ((get_ij(n,numrow,numcol)[0] == get_ij(s,numrow,numcol)[0]) and (get_ij(n,numrow,numcol)[1] < get_ij(s,numrow,numcol)[1]))]
            for n in ymin_neighbors:
              cells_per_subset[n] += 1
          
          #Checking the ymax bound.
          add_cell,add_boundary_cell = check_add_cell(ymax_bound,polygon,current_verts)
          if add_boundary_cell:
            bdy_cells_per_subset[s][3] += 1
          if add_cell:
            #Get the ymin neighbors of the subset.
            ymax_neighbors = [n for n in neighbors if ((get_ij(n,numrow,numcol)[0] == get_ij(s,numrow,numcol)[0]) and (get_ij(n,numrow,numcol)[1] > get_ij(s,numrow,numcol)[1]))]
            for n in ymax_neighbors:
              cells_per_subset[n] += 1
             
          
      
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
