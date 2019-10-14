import sys
sys.path.append(r'C:\Users\tghad\Documents\GitHub\TarekGhaddarGraduateWork\sweep_optimizer\3d')
from optimizer import create_opt_cut_suite_3d
import numpy as np

gxmin = 0.0
gxmax = 60.96
gymin = 0.0
gymax = 60.96
gzmin = 0.0
gzmax = 146.05
numrow = 5
numcol = 5
numplane = 5

points = np.genfromtxt("im1_points")
z_cuts,x_cuts,y_cuts = create_opt_cut_suite_3d(points,gxmin,gxmax,gymin,gymax,gzmin,gzmax,numcol,numrow,numplane)