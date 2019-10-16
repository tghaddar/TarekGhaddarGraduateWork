import numpy as np
import matplotlib.pyplot as plt



num_subsets = [2,3,4,5,6,7,8,9,10]
max_time_reg = np.genfromtxt("more_sparse_reg_data")
max_time_lb = np.genfromtxt("more_sparse_lb_data")
max_time_lbd = np.genfromtxt("more_sparse_lb_data")
max_time_opt = np.genfromtxt("more_sparse_best_data")
plt.figure()
plt.grid(True,axis='y')
plt.xlabel(r'$\sqrt{\rm{Number\ of\ Subsets}}$')
plt.ylabel("TTS (s)")
plt.plot(num_subsets,max_time_reg,'--bx',label="Reg")
plt.plot(num_subsets,max_time_lb,'--g*',label="LB")
plt.plot(num_subsets,max_time_lbd,'--yD',label="LBD")
plt.plot(num_subsets,max_time_opt,'--ro',label="Bin")
plt.legend(loc="best")
plt.savefig("../../figures/more_sparse_best_comp.pdf")