import math
import os
import random
from IPython.core.debugger import Tracer
import matplotlib.pyplot as plt

#Find item in list of lists
def find_cut(subset_boundaries, cut):
  numrows = len(subset_boundaries)
  
  location = []
  for i in range(0, numrows):
    try:
      column = subset_boundaries[i].index(cut)
    except ValueError:
      continue

    location.append(i)

  return location

def find_ymin(subset_boundaries, cut, column):
  for i in range(0,len(subset_boundaries)):
    #The column this subset is in.
    current_column = int(i/(N_x+1))
    if (current_column == column):
      if (subset_boundaries[i][2] == cut):
        return i

def find_ymax(subset_boundaries,cut,column):
  for i in range(0,len(subset_boundaries)):
    #The column this subset is in.
    current_column = int(i/(N_x+1))
    if (current_column == column):
      if (subset_boundaries[i][3] == cut):
        return i

#Number of cuts in x-direction
N_x = 2
#Number of cuts in y-direction
N_y = 2

#Number of subsets
num_subsets = (N_x+1)*(N_y+1)


#Global X minimum
global_x_min = 0.0
#Global X maximum
global_x_max = 10.0

#Global Y minimum
global_y_min = 0.0
global_y_max = 10.0

#Stores x-cuts
#x_cuts = [3.0, 7.0]
x_cuts = []
#2D array that wills store the y_cuts for each y_column.
y_cuts = []
#y_cuts = [[2.0,6.0],[1.0,8.0], [0.5,5.0]]

#Populating x_cuts with uniformly distributed random values.
for i in range(0,N_x):
  cut = random.uniform(global_x_min,global_x_max)
  x_cuts.append(cut)
  
  #For each column, we need uniformly distributed random values of y_cuts.
  y_cuts_column = []

  for j in range(0,N_y):
    cut_y = random.uniform(global_y_min,global_y_max)
    y_cuts_column.append(cut_y)

  y_cuts_column.sort()

  y_cuts.append(y_cuts_column)

#Last column
y_cuts_column = []
for i in range(0,N_y):
  cut_y = random.uniform(global_y_min, global_y_max)
  y_cuts_column.append(cut_y)

y_cuts_column.sort()
y_cuts.append(y_cuts_column)

x_cuts.sort()

print(x_cuts)
print(y_cuts)

#Storing the x_min, x_max, y_min, y_max. 
global_subset_boundary = []
for i in range(0,num_subsets):
  subset_boundary = []
  
  #Subset ID
  ss_id = i

  current_column = int(ss_id/(N_x+1))
  current_row = 0
  if (current_column == 0):
    current_row = ss_id
  else:
    current_row = int(ss_id - current_column*(N_y+1))

  x_min = 0
  x_max = 0
  y_min = 0
  y_max = 0

  #Getting x_min, x_max, y_min, y_max of current subset
  if (current_column == 0):
    x_min = global_x_min
    x_max = x_cuts[current_column]
    
    if (current_row == 0):
      y_min = global_y_min
      y_max = y_cuts[current_column][current_row]
    elif (current_row == N_y):
      y_max = global_y_max
      y_min = y_cuts[current_column][current_row-1]
    else:
      y_min = y_cuts[current_column][current_row-1]
      y_max = y_cuts[current_column][current_row]


  elif (current_column == N_x):
    x_max = global_x_max
    x_min = x_cuts[current_column-1]
    
    if (current_row == 0):
      y_min = global_y_min
      y_max = y_cuts[current_column][current_row]
    elif (current_row == N_y):
      y_max = global_y_max
      y_min = y_cuts[current_column][current_row-1]
    else:
      y_min = y_cuts[current_column][current_row-1]
      y_max = y_cuts[current_column][current_row]
  
  else:
    x_min = x_cuts[current_column-1]
    x_max = x_cuts[current_column]
    
    if (current_row == 0):
      y_min = global_y_min
      y_max = y_cuts[current_column][current_row]
    elif (current_row == N_y):
      y_max = global_y_max
      y_min = y_cuts[current_column][current_row-1]
    else:
      y_min = y_cuts[current_column][current_row-1]
      y_max = y_cuts[current_column][current_row]

  subset_boundary = [x_min, x_max, y_min, y_max]
  #print("Row: ", current_row)
  #print("Col: ", current_column)
  #print(subset_boundary)
  global_subset_boundary.append(subset_boundary)

#For each subset, we determine its neighbors.
#We store it in a simple 2D array for now.
global_neighbor_map = []
for i in range(0,num_subsets):
  #Local neighbor map for this subset
  local_neighbor_map = []
  current_column = int(i/(N_x+1))
  current_row = 0
  if (current_column == 0):
    current_row = ss_id
  else:
    current_row = int(ss_id - current_column*(N_y+1))
  
  
  x_min = global_subset_boundary[i][0]
  x_max = global_subset_boundary[i][1]
  y_min = global_subset_boundary[i][2]
  y_max = global_subset_boundary[i][3]
  
  #If the current subset is in the first column, we only need to look to our right neighbor column
  if (current_column == 0):
    #We pull the y_cuts for the second column
    second_column_y_cuts = y_cuts[1]
    subsets = int(-1)
    for j in range(0,N_y):
      cut = second_column_y_cuts[j]
      #Check if this y_cut is less than or equal to the maximum y boundary of this boundary.
      if (cut < y_max):
        #Check if this y_cut is greater than the minimum y boundary of this subset.
        if (cut > y_min):
          #The subsets that share this cut are added as neighbors.
          subsets = find_cut(global_subset_boundary, cut)
          for s in range(0,len(subsets)):
            local_neighbor_map.append(subsets[s])
        elif (cut == y_min):
          subsets = find_ymin(global_subset_boundary,cut,current_column+1)
          local_neighbor_map.append(subsets)
        else:
          sub = int(N_y)
          sub = sub - 1
          if (cut == second_column_y_cuts[sub]):
            subsets = find_ymin(global_subset_boundary,cut,current_column+1)
            local_neighbor_map.append(subsets)
      elif( cut == y_max):
        #Subset that has this cut as it's ymax is a neighbor
        subsets = find_ymax(global_subset_boundary,cut,current_column+1)
        local_neighbor_map.append(subsets)
      #Cut is greater than maximum y boundary of this subset
      else:
        #Check if we're looking at the first cut line in this column
        if (j == 0):
          subsets = find_ymax(global_subset_boundary,second_column_y_cuts[j],current_column+1)
          local_neighbor_map.append(subsets)
        elif (second_column_y_cuts[j-1] < y_max):
          subsets = find_ymax(global_subset_boundary,cut,current_column+1)
          local_neighbor_map.append(subsets)
  
  #We're only concerned with neighbors to the left
  elif (current_column == N_x):
    left_neighbor_cuts = y_cuts[N_y-1]
    for j in range(0, N_y):
      cut = left_neighbor_cuts[j]
      if (cut < y_max):
        if (cut > y_min):
          subsets = find_cut(global_subset_boundary, cut)
          for s in range(0,len(subsets)):
            local_neighbor_map.append(subsets[s])
        elif(cut == y_min):
          subsets = find_ymin(global_subset_boundary,cut,current_column-1)
          local_neighbor_map.append(subsets)
        else:
          sub = int(N_y)
          sub = sub - 1
          if (cut == left_neighbor_cuts[sub]):
            subsets = find_ymin(global_subset_boundary,cut,current_column-1)
            local_neighbor_map.append(subsets)
      
      elif(cut == y_max):
        subsets = find_ymax(global_subset_boundary,cut,current_column-1)
        local_neighbor_map.append(subsets)
      else:
        if j == 0:
          subsets = find_ymax(global_subset_boundary,left_neighbor_cuts[j],current_column-1)
          local_neighbor_map.append(subsets)
        elif (left_neighbor_cuts[j-1] < y_max):
          subsets = find_ymax(global_subset_boundary,cut,current_column-1)
          local_neighbor_map.append(subsets)
  #Current subset is in interior column, we have to check both right and left neighbors
  else:
    left_neighbor_cuts = y_cuts[current_column-1];
    right_neighbor_cuts = y_cuts[current_column+1];

    for j in range(0,N_y):
      left_cut = left_neighbor_cuts[j]
      right_cut = right_neighbor_cuts[j]

      if (left_cut < y_max):
        if (left_cut > y_min):
          #The subsets that share this cut are added as neighbors.
          subsets = find_cut(global_subset_boundary, left_cut)
          for s in range(0,len(subsets)):
            local_neighbor_map.append(subsets[s])

        elif (left_cut == y_min):
          subsets = find_ymin(global_subset_boundary,left_cut,current_column-1)
          local_neighbor_map.append(subsets)
        else:
          sub = int(N_y)
          sub = sub - 1
          if (left_cut == left_neighbor_cuts[sub]):
            subsets = find_ymin(global_subset_boundary,left_cut,current_column-1)
            local_neighbor_map.append(subsets)
      elif(left_cut == y_max):
        subsets = find_ymax(global_subset_boundary,left_cut,current_column-1)
        local_neighbor_map.append(subsets)
      else:
        if j == 0:
          subsets = find_ymax(global_subset_boundary,left_neighbor_cuts[j],current_column-1)
          local_neighbor_map.append(subsets)
        elif (left_neighbor_cuts[j-1] < y_max):
          subsets = find_ymax(global_subset_boundary,left_cut,current_column-1)
          local_neighbor_map.append(subsets)

      if (right_cut < y_max):
        if (right_cut > y_min):
          #The subsets that share this cut are added as neighbors.
          subsets = find_cut(global_subset_boundary, right_cut)
          for s in range(0,len(subsets)):
            local_neighbor_map.append(subsets[s])

        elif (right_cut == y_min):
          subsets = find_ymin(global_subset_boundary,right_cut,current_column+1)
          local_neighbor_map.append(subsets)
        else:
          sub = int(N_y)
          sub = sub - 1
          if (right_cut == right_neighbor_cuts[sub]):
            subsets = find_ymin(global_subset_boundary,right_cut,current_column+1)
            local_neighbor_map.append(subsets)
      elif(right_cut == y_max):
        subsets = find_ymax(global_subset_boundary,right_cut,current_column+1)
        local_neighbor_map.append(subsets)
      else:
        if j == 0:
          subsets = find_ymax(global_subset_boundary,right_neighbor_cuts[j],current_column+1)
          local_neighbor_map.append(subsets)
        elif (right_neighbor_cuts[j-1] < y_max):
          subsets = find_ymax(global_subset_boundary,right_cut,current_column+1)
          local_neighbor_map.append(subsets)
  
  local_neighbor_map = sorted(set(local_neighbor_map))
  global_neighbor_map.append(local_neighbor_map)


print(global_neighbor_map)

plt.figure(1)

subset_centers = []

for i in range(0,num_subsets):
  subset_boundary = global_subset_boundary[i]
  xmin = subset_boundary[0]
  xmax = subset_boundary[1]
  ymin = subset_boundary[2]
  ymax = subset_boundary[3]

  center_x = (xmin+xmax)/2
  center_y = (ymin+ymax)/2

  subset_centers.append([center_x, center_y])

  x = [xmin, xmax, xmax, xmin, xmin]
  y = [ymin, ymin, ymax, ymax, ymin]

  plt.plot(x,y,'b')


for i in range (0, num_subsets):

  neighbors = global_neighbor_map[i]

  for j in range(0, len(neighbors)):
    n = neighbors[j]
    
    x = [subset_centers[i][0], subset_centers[n][0]]
    y = [subset_centers[i][1], subset_centers[n][1]]
    print(x)
    print(y)
    plt.plot(x,y,'r-o')

plt.savefig("adjacency_plot.png")


