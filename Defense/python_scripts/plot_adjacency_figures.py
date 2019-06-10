import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
from mesh_processor import create_2d_cuts
from build_global_subset_boundaries import build_global_subset_boundaries
from sweep_solver import plot_subset_boundaries_2d
plt.close("all")


numrow = 2
numcol = 2
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
Q0[-1] = [8,0]

#A dictionary for node positions for quadrant 1.
Q1 = {}
Q1[-2] = [-2,0]
Q1[1] = [0,0]
Q1[0] = [2,-1]
Q1[3] = [2,1]
Q1[2] = [4,0]
Q1[-1] = [8,0]

#A dictionary for node positions for quadrant 2.
Q2 = {}
Q2[-2] = [-2,0]
Q2[2] = [0,0]
Q2[0] = [2,-1]
Q2[3] = [2,1]
Q2[1] = [4,0]
Q2[-1] = [8,0]

#A dictionary for node positions for quadrant 3.
Q3 = {}
Q3[-2] = [-2,0]
Q3[3] = [0,0]
Q3[1] = [2,1]
Q3[2] = [2,-1]
Q3[0] = [4, 0]
Q3[-1] = [8,0]


