import numpy as np
from utilities import get_ss_id_3d

def get_ijk_pdt(ss_id,numrow,numcol,numplane):
  k = int(ss_id%numplane)
  j = int((ss_id-k)/numplane) % numrow
  i = int((int((ss_id-k)/numplane - j))/numrow)
  
  return i,j,k

def renumber_subsets(num_subsets,numrow,numcol,numplane):
  
  renumbering = [None]*num_subsets
  
  for s in range(0,num_subsets):
    i,j,k = get_ijk_pdt(s,numrow,numcol,numplane)
    ss_id = get_ss_id_3d(i,j,k,numrow,numcol,numplane)
    renumbering[s] = ss_id
    
  return renumbering

numrow = 3
numcol = 3
numplane = 3
#The number of subsets.
num_subsets = numrow*numcol*numplane

renumbering = renumber_subsets(num_subsets,numrow,numcol,numplane)

subset_data = {}
num_octants = 8
octant_data = {}
combos = [1,-1]
for sign1 in combos:
  for sign2 in combos:
    for sign3 in combos:
      sign = (sign1,sign2,sign3)
      octant_data[sign] = []


for n in range(0,num_subsets):
  f = open(str(n),'r')
  contents = f.readlines()
  num_lines = len(contents)
  for line in range(0,num_lines):
    contents[line] = contents[line].split()
    del contents[line][0:2]
    del contents[line][4]
    del contents[line][3:5]
    contents[line][0] = contents[line][0][1:]
    contents[line][0] = contents[line][0][:-1]
    contents[line][1] = contents[line][1][:-1]
    contents[line][2] = contents[line][2][:-1]
    contents[line][3] = "stage: "
  f.close()
  new_number = renumbering[n]
  subset_data[new_number] = contents

for n in range(0,num_subsets):
  
  data = subset_data[n]
  
  for o in range(0,num_octants):
    octant = data[o]
    omegax = np.sign(float(octant[0]))
    omegay = np.sign(float(octant[1]))
    omegaz = np.sign(float(octant[2]))
    
    omega = (omegax,omegay,omegaz)
    stage = octant[3]
    stage_num = int(octant[4])
    stage_data = (stage,stage_num,n)
    octant_data[omega].append(stage_data)

for sign1 in combos:
  for sign2 in combos:
    for sign3 in combos:
      sign = (sign1,sign2,sign3)
      octant_data[sign].sort(key=lambda x:x[1])