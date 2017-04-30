import numpy as np
from numpy import unravel_index
from tabulate import tabulate

#Manipulating the data from the load balancing runs



LBDSameSide = np.loadtxt(open("LBDSameSide.csv","rb"),delimiter = ",")
LBSameSide = np.loadtxt(open("LBSameSide.csv","rb"),delimiter=",")

sameside_improvement = 1.0 - np.divide(LBDSameSide,LBSameSide)

areas = np.array([[10],[1.8],[1.6],[1.4],[1.2],[1],[0.8],[0.6],[0.4],[0.2],[0.1],[0.08],[0.06],[0.05],[0.04],[0.03],[0.02],[0.01]])
areas.reshape(18,1)

a = np.hstack((areas,sameside_improvement))

print(tabulate(a,tablefmt="latex",floatfmt=".3f"))

#np.savetxt("lbd_opp.csv",LBDopp)
#np.savetxt("lb_opp.csv",LBopp)
#np.savetxt("opp_improvement.csv",opp_improvement)