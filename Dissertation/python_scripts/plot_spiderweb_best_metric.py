import numpy as np
import matplotlib.pyplot as plt
plt.close("all")
subsets = [2,3,4,5,6,7,8,9,10]
regular_metrics = np.genfromtxt("regular_metrics.txt")
lb_metrics = np.genfromtxt("lb_metrics.txt")
lbd_metrics = np.genfromtxt("lbd_metrics.txt")
best_metrics = np.genfromtxt("spiderweb_metric_best")

plt.figure()
plt.grid(True,axis='y')
plt.xlabel(r'$\sqrt{\rm{Number\ of\ Subsets}}$')
plt.ylabel('f')
plt.plot(subsets,regular_metrics,'r-o',label="Reg")
plt.plot(subsets,lb_metrics,'-o',label="LB")
plt.plot(subsets,lbd_metrics,'-o',label="LBD")
plt.plot(subsets,best_metrics,'-o',label="Bin")
plt.legend(loc='best')
plt.savefig("../../figures/spiderweb_metric_study.pdf")