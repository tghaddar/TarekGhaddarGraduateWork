import numpy as np

num_cells = 296994

f = open("points",'r')
points = f.readlines()
points = [x.strip() for x in points]
num_points = int(points[0])
points.pop(0)

#Stripping the data down to the essentials.
for p in range(0,num_points):
  line = points[p]
  line = line[1:]
  line = line[:-1]
  line = line.split()
  line = [float(x) for x in line]
  points[p] = line

f.close()
np.savetxt("../../../Dissertation/python_scripts/im1_points",points)

#Face data
f = open("faces",'r')
faces_file = f.readlines()
faces_file = [x.strip() for x in faces_file]
num_faces = int(faces_file[0])
faces_file.pop(0)
face_centers = [None]*num_faces
for face in range(0,num_faces):
  line = faces_file[face]
  line = line[2:]
  line = line[:-1]
  line = line.split()
  line = [int(x) for x in line]
  num_face_pts = len(line)
  xcenter = 0.0
  ycenter = 0.0
  zcenter = 0.0
  for p in line:
    xcenter += points[p][0]
    ycenter += points[p][1]
    zcenter += points[p][2]
  
  center = [xcenter/num_face_pts,ycenter/num_face_pts,zcenter/num_face_pts]
  face_centers[face] = center
  
f.close()

#Stores the cells
cells = [[] for i in range(0,num_cells)]

#Owners
f = open("owner",'r')
owner_file = f.readlines()
owner_file.pop(0)
owner_file=[int(x.strip()) for x in owner_file]

for face in range(0,num_faces):
  cell = owner_file[face]
  #Appending the face to the cell.
  cells[cell].append(face)

f.close()
#Neighbor file
f = open("neighbour",'r')
neighbors = f.readlines()
neighbors = [int(x.strip()) for x in neighbors]
num_interior_face = int(neighbors[0])
neighbors.pop(0)
for n in range(0,num_interior_face):
  neighbor = neighbors[n]
  cells[neighbor].append(n)
f.close()

#Calculating cells centers.
cell_centers = np.empty([3,num_cells])
for c in range(0,num_cells):
  cell = cells[c]
  num_cell_faces = len(cell)
  print(c,num_cell_faces)
  xcenter = 0.0
  ycenter = 0.0
  zcenter = 0.0
  for face in cell:
    xcenter += face_centers[face][0]
    ycenter += face_centers[face][1]
    zcenter += face_centers[face][2]
  
  cell_centers[0][c] = xcenter/num_cell_faces
  cell_centers[1][c] = ycenter/num_cell_faces
  cell_centers[2][c] = zcenter/num_cell_faces

np.savetxt("im1_cell_centers",cell_centers.T)
