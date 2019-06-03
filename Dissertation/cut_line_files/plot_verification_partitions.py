import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')

from build_global_subset_boundaries import build_global_subset_boundaries
import numpy as np
import matplotlib.pyplot as plt

subsets = [2,3,4,5,6,7,8,9,10]
partition_types = ['regular','mild_random','random','worst']
partition_titles = ['Regular', 'Mildly Random', 'Random', '"Worst"']
counter = 0
for partition_type in partition_types:
  for s in subsets:
    print(s)
    
    x_file_name = "x_cuts_"+str(s)+"_"+partition_type+".csv"
    y_file_name = "y_cuts_"+str(s)+"_"+partition_type+".csv"
    
    x_cuts = np.genfromtxt(x_file_name,delimiter=",")
    y_cuts = np.genfromtxt(y_file_name,delimiter=",")
    
    plot_name = str(s) + " " + partition_type
    plt.figure(plot_name)
    plt.title(str(s) + "x" + str(s) + " Subsets with " + partition_titles[counter] + " Partitions")
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    
    subset_boundaries = build_global_subset_boundaries(s-1,s-1,x_cuts,y_cuts)
    
    for i in range(0,len(subset_boundaries)):
    
      subset_boundary = subset_boundaries[i]
      xmin = subset_boundary[0]
      xmax = subset_boundary[1]
      ymin = subset_boundary[2]
      ymax = subset_boundary[3]
        
      x = [xmin, xmax, xmax, xmin,xmin]
      y = [ymin, ymin, ymax, ymax,ymin]
    
      plt.plot(x,y,'b')
    
    plt.savefig(str(s)+"_"+partition_type+".pdf")
    plt.close()
  counter += 1
