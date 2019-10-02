import numpy as np
import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
from sweep_solver import optimized_tts_numerical,unpack_parameters
from mesh_processor import create_2d_cuts
from optimizer import create_parameter_space,create_bounds,create_constraints,get_column_cdf,create_opt_cut_suite,get_highest_jumps
from scipy.optimize import basinhopping, minimize
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
plt.close("all")
#Communication time per double
t_comm = 4.47e-09
#The number of bytes to communicate per subset.
#The message latency time.
m_l = 50
latency = 4110.0e-09
#Solve time per cell..
Tc = 1208.383e-09
upc = 4.0
upbc = 2.0
Twu = 147.0754e-09
Tm = 65.54614e-09
Tg = 175.0272e-09
mcff = 1.181
machine_parameters = (Twu,Tc,Tm,Tg,upc,upbc,mcff,t_comm,latency,m_l)
num_angles = 1
Am = 36
unweighted = True
Ay = 1

points = np.genfromtxt("unbalanced_pins_sparse_centroid_data").T
verts = np.genfromtxt("unbalanced_pins_sparse_vert_data")
f = open("sparse_pins_cell_verts",'r')
sparse_pins_cell_data = [line.split() for line in f]
for i in range(0,len(sparse_pins_cell_data)):
  sparse_pins_cell_data[i] = [int(x) for x in sparse_pins_cell_data[i]]

gxmin = 0.0
gxmax = 10.0
gymin = 0.0
gymax = 10.0

numrows = [2,3,4,5,6,7,9,10]
numcols = [2,3,4,5,6,7,9,10]
max_times = []

for i in range(0,len(numrows)):
  numcol = numcols[i]
  numrow = numrows[i]
  x_file = "spiderweb_lb_cut_line_data/cut_line_data_x_"+str(numrow)
  y_file = "spiderweb_lb_cut_line_data/cut_line_data_y_"+str(numrow)
  x_cuts = np.genfromtxt(x_file)
  y_cuts = np.genfromtxt(y_file)
  params = create_parameter_space(x_cuts,y_cuts,numrow,numcol)
  add_cells = True
  
  max_times.append( optimized_tts_numerical(params,sparse_pins_cell_data,verts,gxmin,gxmax,gymin,gymax,numrow,numcol,machine_parameters,num_angles,Am,Ay,add_cells,unweighted))
  
pdt_data = np.genfromtxt("spiderweb_lb_sweep_data.txt")
pdt_data = np.reshape(pdt_data,(8,10))
pdt_data_median = np.empty(8)
percent_diff = np.empty(8)
for i in range(0,8):
  median = np.median(pdt_data[i])
  tts = max_times[i]
  pdt_data_median[i] = median
  percent_diff[i] = abs(tts-median)/median*100
  
plt.figure()
plt.xlabel(r'$\sqrt{\rm{Number\ of\ Subsets}}$')
plt.ylabel("Sweep Time (s)")
plt.plot(numrows,pdt_data_median,'-o',label="PDT")
plt.plot(numrows,max_times,'-o',label="TTS")
plt.legend(loc="best")
plt.savefig("../../figures/spiderweb_lb_pdtvtts.pdf")

f = open("spiderweb_lb_percent_diff.txt",'w')
f.write("\\textbf{$\sqrt{\\text{Num Subsets}}$} & \\bf PDT vs. TTS \\\ \hline \n")
for i in range(0,len(numrows)):
  f.write( str(numrows[i])+'&'+str(np.round(percent_diff[i],2))+'\%')
  if i < len(numrows)-1:
    f.write("\\\ \hline \n")
  else:
    f.write("\n")
f.close()