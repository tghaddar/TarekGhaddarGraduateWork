import numpy as np

gxmin = 0.0
gxmax = 60.96
gymin = 0.0
gymax = 60.96
gzmin = 0.0
gzmax = 146.05

#The machine parameters.
#Communication time per double
t_comm = 4.47e-09
#The number of bytes to communicate per subset.
#The message latency time.
m_l = 1
latency = 4110.0e-09
#Solve time per unknown.
t_u = 450.0e-09
upc = 4.0
upbc = 2.0
machine_params = (t_u,upc,upbc,t_comm,latency,m_l)

num_angles = 1
unweighted = True