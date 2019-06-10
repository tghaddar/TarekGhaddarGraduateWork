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

