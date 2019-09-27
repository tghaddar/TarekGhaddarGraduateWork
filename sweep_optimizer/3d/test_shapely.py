from shapely.geometry import MultiPoint,LineString
from mesh_processor import check_add_cell
import numpy as np

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
    return is_nat_boundary
  elif (num_intersections < 1):
    raise Exception("Less than one intersection. Shouldn't be possible at this point.")
  else:
    ctr  = 0
    for j in range(0,num_intersections):
      if intersect_coords[j] in polygon.exterior.coords[:]:
        ctr += 1
    if ctr == 2:    
      is_colinear = check_colinear(bound,bound_intersections)
      if is_colinear:
        is_nat_boundary = True
    else:
      return is_nat_boundary

      
  
  return is_nat_boundary

def check_add_cell(polygon,int_bounds,bounds):
  
  add_cell = [False,False,False,False]
  #Checking for a natural boundary for the polygon.
  for i in int_bounds:
    bound = bounds[i]
    #Checking if this boundary intersection is a natural boundary. 
    is_nat_boundary = check_nat_boundary(polygon,bound)
    if is_nat_boundary == False:
      add_cell[i] = True
     
    
    
  return add_cell

xmin = 0.0
xmax = 1.0
ymin = 0.0
ymax = 1.0

xmin_bound = LineString([(xmin,ymin), (xmin,ymax)])
xmax_bound = LineString([(xmax,ymin), (xmax,ymax)])
ymin_bound = LineString([(xmin,ymin), (xmax,ymin)])
ymax_bound = LineString([(xmin,ymax), (xmax,ymax)])

bounds = [xmin_bound,xmax_bound,ymin_bound,ymax_bound]
add_cell = [False,False,False,False]

    

subset = MultiPoint([(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)]).convex_hull

bound = LineString([(0,0),(0,1)])
current_verts = [(0.0,0.2),(0.0,0.6),(0.5,0.6),(1.5,0.2)   ]
polygon = MultiPoint(current_verts).convex_hull


intersect = polygon.intersection(subset)
intersection_coords = intersect.exterior.coords[:]
num_intersections = len(intersection_coords)
int_bounds = []
if num_intersections > 1:
  int_bounds = which_bounds(intersect,bounds)
  add_cell = check_add_cell(polygon,int_bounds,bounds)

#is_nat_boundary = False
#ctr = 0
#for i in range(0,len(intersection_coords)):
#  if intersection_coords[i] in current_verts:
#    ctr += 1
#    
#if ctr == 2:
#  is_nat_boundary = True

#add_cell,add_boundary_cell = check_add_cell(bound,current_verts)