import numpy as np
import matplotlib.pyplot as plt
plt.close("all")
plt.figure()


num_subsets = [2,3,4,5,6,7,8,9,10]
opt = np.genfromtxt("spiderweb_opt_times.csv")
reg = np.genfromtxt("spiderweb_regular_times.csv")
plt.plot(num_subsets,opt,'-o',label="Opt")
plt.plot(num_subsets,reg,'-o',label="Reg")

num_subsets = [2,3,4,5,6,7,9,10]
lb = np.genfromtxt("spiderweb_lb_times.csv")
plt.plot(num_subsets,lb,'-o',label="LB")

num_subsets = [2,3,4,6,7,9,10]
lbd = np.genfromtxt("spiderweb_lbd_times.csv")
plt.plot(num_subsets,lbd,'-o',label="LBD")
plt.legend(loc='best')

plt.ylabel("Sweep Time (s)")
plt.xlabel(r'$\sqrt{\rm{Number\ of\ Subsets}}$')
plt.savefig("../../figures/unbalanced_pins_opt_comparison.pdf")
