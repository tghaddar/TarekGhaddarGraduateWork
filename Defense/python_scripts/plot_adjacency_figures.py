import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
#sys.path.append(r'C:\Users\tghad\Documents\GitHub\TarekGhaddarGraduateWork\sweep_optimizer\3d')
from mesh_processor import create_2d_cuts
from build_global_subset_boundaries import build_global_subset_boundaries
from sweep_solver import plot_subset_boundaries_2d,add_edge_cost
from sweep_solver import make_edges_universal,pipeline_offset
from build_adjacency_matrix import build_adjacency,build_graphs 
import networkx as nx
from copy import deepcopy
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
graphs = build_graphs(adjacency_matrix,numrow,numcol,2)

plt.figure("Quadrant 0")
plt.title("Quadrant 0 Graph")
#edge_labels_1 = nx.get_edge_attributes(graphs[0])
nx.draw(graphs[0],Q0,with_labels = True,node_color='red')
plt.savefig("../../figures/q0_preweight.pdf")
plt.close()

plt.figure("Quadrant 3")
plt.title("Quadrant 3 Graph")
#edge_labels_1 = nx.get_edge_attributes(graphs[0])
nx.draw(graphs[3],Q3,with_labels = True,node_color='red')
plt.savefig("../../figures/q3_preweight.pdf")
plt.close()

plt.figure("Quadrant 1")
plt.title("Quadrant 1 Graph")
#edge_labels_1 = nx.get_edge_attributes(graphs[0])
nx.draw(graphs[1],Q1,with_labels = True,node_color='red')
plt.savefig("../../figures/q1_preweight.pdf")
plt.close()

plt.figure("Quadrant 2")
plt.title("Quadrant 2 Graph")
#edge_labels_1 = nx.get_edge_attributes(graphs[0])
nx.draw(graphs[2],Q2,with_labels = True,node_color='red')
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

#The machine parameters.
#Communication time per double
t_comm = 4.47e-02
#The number of bytes to communicate per subset.
#The message latency time.
m_l = 1
latency = 4110.0e-02
#Solve time per unknown.
t_u = 450.0e-02
upc = 4.0
upbc = 2.0
machine_params = (t_u,upc,upbc,t_comm,latency,m_l)

#Dummy values for the purpose of this test case. 
cells_per_subset = [1 for n in range(0,num_subsets)]
bdy_cells_per_subset = [[1,1] for n in range(0,num_subsets)]
graphs,time_to_solve = add_edge_cost(graphs,bounds,cells_per_subset,bdy_cells_per_subset,machine_params,numrow,numcol,True)

plt.figure("Quadrant 0")
plt.title("Quadrant 0 Graph")
edge_labels_1 = nx.get_edge_attributes(graphs[0],'weight')
nx.draw(graphs[0],Q0,with_labels = True,node_color='red')
nx.draw_networkx_edge_labels(graphs[0],Q0,edge_labels=edge_labels_1,font_size=12)
plt.savefig("../../figures/q0_postweight.pdf")
plt.close()

plt.figure("Quadrant 1")
plt.title("Quadrant 1 Graph")
edge_labels_1 = nx.get_edge_attributes(graphs[1],'weight')
nx.draw(graphs[1],Q1,with_labels = True,node_color='red')
nx.draw_networkx_edge_labels(graphs[1],Q1,edge_labels=edge_labels_1,font_size=12)
plt.savefig("../../figures/q1_postweight.pdf")
plt.close()

plt.figure("Quadrant 2")
plt.title("Quadrant 2 Graph")
edge_labels_1 = nx.get_edge_attributes(graphs[2],'weight')
nx.draw(graphs[2],Q2,with_labels = True,node_color='red')
nx.draw_networkx_edge_labels(graphs[2],Q2,edge_labels=edge_labels_1,font_size=12)
plt.savefig("../../figures/q2_postweight.pdf")
plt.close()

plt.figure("Quadrant 3")
plt.title("Quadrant 3 Graph")
edge_labels_1 = nx.get_edge_attributes(graphs[3],'weight')
nx.draw(graphs[3],Q3,with_labels = True,node_color='red')
nx.draw_networkx_edge_labels(graphs[3],Q3,edge_labels=edge_labels_1,font_size=12)
plt.savefig("../../figures/q3_postweight.pdf")
plt.close()



graphs = pipeline_offset(graphs,2,time_to_solve)
#POST PIPELINENING
plt.figure("Quadrant 0")
plt.title("Quadrant 0 Graph")
edge_labels_1 = nx.get_edge_attributes(graphs[0],'weight')
nx.draw(graphs[0],Q0,with_labels = True,node_color='red')
nx.draw_networkx_edge_labels(graphs[0],Q0,edge_labels=edge_labels_1,font_size=12)
plt.savefig("../../figures/q0_postpipeline.pdf")
plt.close()

plt.figure("Quadrant 1")
plt.title("Quadrant 1 Graph")
edge_labels_1 = nx.get_edge_attributes(graphs[1],'weight')
nx.draw(graphs[1],Q1,with_labels = True,node_color='red')
nx.draw_networkx_edge_labels(graphs[1],Q1,edge_labels=edge_labels_1,font_size=12)
plt.savefig("../../figures/q1_postpipeline.pdf")
plt.close()

plt.figure("Quadrant 2")
plt.title("Quadrant 2 Graph")
edge_labels_1 = nx.get_edge_attributes(graphs[2],'weight')
nx.draw(graphs[2],Q2,with_labels = True,node_color='red')
nx.draw_networkx_edge_labels(graphs[2],Q2,edge_labels=edge_labels_1,font_size=12)
plt.savefig("../../figures/q2_postpipeline.pdf")
plt.close()

plt.figure("Quadrant 3")
plt.title("Quadrant 3 Graph")
edge_labels_1 = nx.get_edge_attributes(graphs[3],'weight')
nx.draw(graphs[3],Q3,with_labels = True,node_color='red')
nx.draw_networkx_edge_labels(graphs[3],Q3,edge_labels=edge_labels_1,font_size=12)
plt.savefig("../../figures/q3_postpipeline.pdf")
plt.close()

#Second Angle.
plt.figure("Quadrant 0")
plt.title("Quadrant 0 Angle 2")
edge_labels_1 = nx.get_edge_attributes(graphs[4],'weight')
nx.draw(graphs[4],Q0,with_labels = True,node_color='red')
nx.draw_networkx_edge_labels(graphs[4],Q0,edge_labels=edge_labels_1,font_size=12)
plt.savefig("../../figures/q4_postpipeline.pdf")
plt.close()

plt.figure("Quadrant 1")
plt.title("Quadrant 1 Angle 2")
edge_labels_1 = nx.get_edge_attributes(graphs[5],'weight')
nx.draw(graphs[5],Q1,with_labels = True,node_color='red')
nx.draw_networkx_edge_labels(graphs[5],Q1,edge_labels=edge_labels_1,font_size=12)
plt.savefig("../../figures/q5_postpipeline.pdf")
plt.close()

plt.figure("Quadrant 2")
plt.title("Quadrant 2 Angle 2")
edge_labels_1 = nx.get_edge_attributes(graphs[6],'weight')
nx.draw(graphs[6],Q2,with_labels = True,node_color='red')
nx.draw_networkx_edge_labels(graphs[6],Q2,edge_labels=edge_labels_1,font_size=12)
plt.savefig("../../figures/q6_postpipeline.pdf")
plt.close()

plt.figure("Quadrant 3")
plt.title("Quadrant 3 Angle 2")
edge_labels_1 = nx.get_edge_attributes(graphs[7],'weight')
nx.draw(graphs[7],Q3,with_labels = True,node_color='red')
nx.draw_networkx_edge_labels(graphs[7],Q3,edge_labels=edge_labels_1,font_size=12)
plt.savefig("../../figures/q7_postpipeline.pdf")
plt.close()


graphs = make_edges_universal(graphs)
plt.figure("Quadrant 0")
plt.title("Quadrant 0 Graph")
edge_labels_1 = nx.get_edge_attributes(graphs[0],'weight')
nx.draw(graphs[0],Q0,with_labels = True,node_color='red')
nx.draw_networkx_edge_labels(graphs[0],Q0,edge_labels=edge_labels_1,font_size=12)
plt.savefig("../../figures/q0_postuniversal.pdf")
plt.close()

plt.figure("Quadrant 1")
plt.title("Quadrant 1 Graph")
edge_labels_1 = nx.get_edge_attributes(graphs[1],'weight')
nx.draw(graphs[1],Q1,with_labels = True,node_color='red')
nx.draw_networkx_edge_labels(graphs[1],Q1,edge_labels=edge_labels_1,font_size=12)
plt.savefig("../../figures/q1_postuniversal.pdf")
plt.close()

plt.figure("Quadrant 2")
plt.title("Quadrant 2 Graph")
edge_labels_1 = nx.get_edge_attributes(graphs[2],'weight')
nx.draw(graphs[2],Q2,with_labels = True,node_color='red')
nx.draw_networkx_edge_labels(graphs[2],Q2,edge_labels=edge_labels_1,font_size=12)
plt.savefig("../../figures/q2_postuniversal.pdf")
plt.close()

plt.figure("Quadrant 3")
plt.title("Quadrant 3 Graph")
edge_labels_1 = nx.get_edge_attributes(graphs[3],'weight')
nx.draw(graphs[3],Q3,with_labels = True,node_color='red')
nx.draw_networkx_edge_labels(graphs[3],Q3,edge_labels=edge_labels_1,font_size=12)
plt.savefig("../../figures/q3_postuniversal.pdf")
plt.close()

#Second angle post universal.
plt.figure("Quadrant 0")
plt.title("Quadrant 0 Angle 2")
edge_labels_1 = nx.get_edge_attributes(graphs[4],'weight')
nx.draw(graphs[4],Q0,with_labels = True,node_color='red')
nx.draw_networkx_edge_labels(graphs[4],Q0,edge_labels=edge_labels_1,font_size=12)
plt.savefig("../../figures/q4_postuniversal.pdf")
plt.close()

plt.figure("Quadrant 1")
plt.title("Quadrant 1 Angle 2")
edge_labels_1 = nx.get_edge_attributes(graphs[5],'weight')
nx.draw(graphs[5],Q1,with_labels = True,node_color='red')
nx.draw_networkx_edge_labels(graphs[5],Q1,edge_labels=edge_labels_1,font_size=12)
plt.savefig("../../figures/q5_postuniversal.pdf")
plt.close()

plt.figure("Quadrant 2")
plt.title("Quadrant 2 Angle 2")
edge_labels_1 = nx.get_edge_attributes(graphs[6],'weight')
nx.draw(graphs[6],Q2,with_labels = True,node_color='red')
nx.draw_networkx_edge_labels(graphs[6],Q2,edge_labels=edge_labels_1,font_size=12)
plt.savefig("../../figures/q6_postuniversal.pdf")
plt.close()

plt.figure("Quadrant 3")
plt.title("Quadrant 3 Angle 2")
edge_labels_1 = nx.get_edge_attributes(graphs[7],'weight')
nx.draw(graphs[7],Q3,with_labels = True,node_color='red')
nx.draw_networkx_edge_labels(graphs[7],Q3,edge_labels=edge_labels_1,font_size=12)
plt.savefig("../../figures/q7_postuniversal.pdf")
plt.close()
