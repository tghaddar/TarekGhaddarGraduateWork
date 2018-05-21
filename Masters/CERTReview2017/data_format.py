import numpy as np
from numpy import unravel_index
from tabulate import tabulate

#Manipulating the data from the load balancing runs



#new = np.loadtxt(open("LBD_opp.csv","rb"),dtype=np.string,delimiter = ",")
#old = np.loadtxt(open("LB_opp.csv","rb"),dtype=np.string,delimiter=",")
new = np.genfromtxt("LBD_same.csv",dtype='str',delimiter=',')
old = np.genfromtxt("LB_same.csv",dtype='str',delimiter=",")

#improvement = 1.0 - np.divide(new,old)

areas = np.array([[10],[1.8],[1.6],[1.4],[1.2],[1],[0.8],[0.6],[0.4],[0.2],[0.1],[0.08],[0.06],[0.05],[0.04],[0.03],[0.02],[0.01]])
areas.reshape(18,1)

a = np.hstack((areas,new))
b = np.hstack((areas,old))

print(tabulate(b,tablefmt="latex"))
print(tabulate(a,tablefmt="latex"))



print(unravel_index(old.argmin(),old.shape))
print(unravel_index(old.argmax(),old.shape))

print(unravel_index(new.argmin(),new.shape))
print(unravel_index(new.argmax(),new.shape))

#print(unravel_index(improvement.argmin(),improvement.shape))
#print(unravel_index(improvement.argmax(),improvement.shape))


#np.savetxt("lbd_opp.csv",LBDopp)
#np.savetxt("lb_opp.csv",LBopp)
#np.savetxt("opp_improvement.csv",opp_improvement)