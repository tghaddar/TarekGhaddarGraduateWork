import numpy as np
import matplotlib.pyplot as plt
import sys
#sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
sys.path.append(r'C:\Users\tghad\Documents\GitHub\TarekGhaddarGraduateWork\sweep_optimizer\3d')
from mesh_processor import create_2d_cuts
from build_global_subset_boundaries import build_global_subset_boundaries
from sweep_solver import plot_subset_boundaries_2d
from build_adjacency_matrix import build_adjacency,build_graphs 
import networkx as nx
from utilities import get_ijk
plt.close("all")


numrow = 2
numcol = 2
numplane = 1
num_subsets = numrow*numcol
xmin = 0.0
xmax = 10.0
ymin = 0.0
ymax = 10.0

x_cuts,y_cuts = create_2d_cuts(xmin,xmax,numcol,ymin,ymax,numrow)
bounds = build_global_subset_boundaries(numcol-1,numrow-1,x_cuts,y_cuts)
plot_subset_boundaries_2d(bounds,num_subsets)


#A dictionary for node positions for quadrant 0.
Q0 = {}
Q0[-2] = [-2,0]
Q0[0] = [0,0]
Q0[1] = [2,1]
Q0[2] = [2,-1]
Q0[3] = [4, 0]
Q0[-1] = [6,0]

#A dictionary for node positions for quadrant 1.
Q1 = {}
Q1[-2] = [-2,0]
Q1[1] = [0,0]
Q1[0] = [2,-1]
Q1[3] = [2,1]
Q1[2] = [4,0]
Q1[-1] = [6,0]

#A dictionary for node positions for quadrant 2.
Q2 = {}
Q2[-2] = [6,0]
Q2[1] = [0,0]
Q2[0] = [2,-1]
Q2[3] = [2,1]
Q2[2] = [4,0]
Q2[-1] = [-2,0]

#A dictionary for node positions for quadrant 3.
Q3 = {}
Q3[-2] = [6,0]
Q3[0] = [0,0]
Q3[1] = [2,1]
Q3[2] = [2,-1]
Q3[3] = [4, 0]
Q3[-1] = [-2,0]

adjacency_matrix = build_adjacency(bounds,numcol-1,numrow-1,y_cuts)
graphs = build_graphs(adjacency_matrix,numrow,numcol,1)

plt.figure("Quadrant 0")
plt.title("Quadrant 0 Graph")
#edge_labels_1 = nx.get_edge_attributes(graphs[0])
nx.draw(graphs[0],Q0,with_labels = True)
plt.savefig("../../figures/q0_preweight.pdf")
plt.close()

plt.figure("Quadrant 3")
plt.title("Quadrant 3 Graph")
#edge_labels_1 = nx.get_edge_attributes(graphs[0])
nx.draw(graphs[3],Q3,with_labels = True)
plt.savefig("../../figures/q3_preweight.pdf")
plt.close()

plt.figure("Quadrant 1")
plt.title("Quadrant 1 Graph")
#edge_labels_1 = nx.get_edge_attributes(graphs[0])
nx.draw(graphs[1],Q1,with_labels = True)
plt.savefig("../../figures/q1_preweight.pdf")
plt.close()

plt.figure("Quadrant 2")
plt.title("Quadrant 2 Graph")
#edge_labels_1 = nx.get_edge_attributes(graphs[0])
nx.draw(graphs[2],Q2,with_labels = True)
plt.savefig("../../figures/q2_preweight.pdf")
plt.close()

#Plotting the "flipped"  numbering. 
plt.figure("flipped adjacency")
subset_centers = []
for i in range(0,num_subsets):

  subset_boundary = bounds[i]
  xmin = subset_boundary[0]
  xmax = subset_boundary[1]
  ymin = subset_boundary[2]
  ymax = subset_boundary[3]

  center_x = (xmin+xmax)/2
  center_y = (ymin+ymax)/2

  subset_centers.append([center_x, center_y])

  x = [xmin, xmax, xmax, xmin,xmin]
  y = [ymin, ymin, ymax, ymax,ymin]

  plt.plot(x,y,'r')
  #Getting the i,j indices of the subset.
  i,j,k = get_ijk(i,numrow,numcol,numplane)
  #The maximum subset in this column when ordered conventionally.
  max_subset = k*num_subsets  + i*numrow + (numrow-1)
  #The new subset id in the flipped ordering.
  new_ss_id = max_subset - j
  plt.text(center_x,center_y,str(new_ss_id))

plt.savefig("../../figures/flipped_subset_layout.pdf")
plt.close()



