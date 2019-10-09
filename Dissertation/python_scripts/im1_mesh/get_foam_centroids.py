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
#np.savetxt("../../../Dissertation/python_scripts/im1_points",points)

#Face data
f = open("faces",'r')
faces_file = f.readlines()
faces_file = [x.strip() for x in faces_file]
num_faces = int(faces_file[0])
faces_file.pop(0)
face_centers = [None]*num_faces
faces = [None]*num_faces
for face in range(0,num_faces):
  line = faces_file[face]
  line = line[2:]
  line = line[:-1]
  line = line.split()
  faces[face] = [int(x) for x in line]
  
f.close()

#Stores the cells
cells = [[] for i in range(0,num_cells)]
cell_2_point = [[] for i in range(0,num_cells)]

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

#Getting connectivity for each cell.
for c in range(0,num_cells):
  cell = cells[c]
  num_cell_faces = len(cell)
  for face in cell:
    points = faces[face]
    for p in points:
      cell_2_point[c].append(int(p))
  
  cell_2_point[c] = list(set(cell_2_point[c]))
    

np.savetxt("im1_cell_vert_data",cell_2_point,fmt='%d')
