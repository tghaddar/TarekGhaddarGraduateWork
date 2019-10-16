import numpy as np
import matplotlib.pyplot as plt
plt.close("all")
plt.figure()
plt.grid(True,axis='y')

num_subsets = [2,3,4,5,6,7,8,9,10]
opt = np.genfromtxt("spiderweb_best_times.csv")
reg = np.genfromtxt("spiderweb_regular_times.csv")
plt.plot(num_subsets,opt,'-ro',label="Bin")
plt.plot(num_subsets,reg,'-bx',label="Reg")

num_subsets = [2,3,4,5,6,7,9,10]
lb = np.genfromtxt("spiderweb_lb_times.csv")
plt.plot(num_subsets,lb,'-g*',label="LB")

num_subsets = [2,3,4,6,7,9,10]
lbd = np.genfromtxt("spiderweb_lbd_times.csv")
plt.plot(num_subsets,lbd,'-yD',label="LBD")
plt.legend(loc='best')

plt.ylabel("TTS (s)")
plt.xlabel(r'$\sqrt{\rm{Number\ of\ Subsets}}$')
plt.savefig("../../figures/unbalanced_pins_best_comparison.pdf")
