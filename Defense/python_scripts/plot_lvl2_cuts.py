import numpy as np
import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
from sweep_solver import plot_subset_boundaries_2d
from build_global_subset_boundaries import build_global_subset_boundaries



x_cuts = np.genfromtxt("x_cuts_lvl2")
y_cuts = np.genfromtxt("y_cuts_lvl2")

boundaries = build_global_subset_boundaries(41,12,x_cuts,y_cuts)

x_cuts_lb = np.genfromtxt("x_cuts_lvl2_lb")
y_cuts_lb = np.genfromtxt("y_cuts_lvl2_lb")

boundaries_lb = build_global_subset_boundaries(41,12,x_cuts_lb,y_cuts_lb)

plot_subset_boundaries_2d(boundaries,42*13,"../../figures/lvl2regularcuts.pdf")
plot_subset_boundaries_2d(boundaries_lb,42*13,"../../figures/lvl2lbcuts.pdf")