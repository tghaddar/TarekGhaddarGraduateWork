import math
import os
import random

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
x_cuts = []
#2D array that wills store the y_cuts for each y_column.
y_cuts = []

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


y_cuts_column = []
for i in range(0,N_y):
  cut_y=random.uniform(global_y_min, global_y_max)
  y_cuts_column.append(cut_y)

y_cuts_column.sort()
y_cuts.append(y_cuts_column)

x_cuts.sort()

print(x_cuts)
print(y_cuts)

#For each subset, we determine its neighbors.
#We store it in a simple 2D array for now.
global_neighbor_map = []
for i in range(0,num_subsets):
  subset_neighbor_map = []
  
  #Subset ID
  ss_id = i

  current_column = ss_id/(N_x+1)
  current_row = 

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

  print("Subset ID: " , ss_id)
  print("Current Row): ", current_row)
  print("Current Col): ", current_column)
  print(x_min,y_min,x_max,y_max)
  #If the current subset is in the first column, we only need to look to our right neighbor column
  if (current_column == 0):
    #We pull the y_cuts for the second column
    second_column_y_cuts = y_cuts[1]

    #Loop over the y_cuts in this column
    #for j in range(0, N_y):
      
