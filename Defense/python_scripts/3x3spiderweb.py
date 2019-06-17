import numpy as np

max_time_opt = np.genfromtxt("solvepersweep_3_opt_fcfs.txt")
max_time_lbd = np.genfromtxt("solvepersweep_3_lbd_fcfs.txt")
max_time_opt_mean = np.mean(max_time_opt)
max_time_lbd_mean = np.mean(max_time_lbd)
max_time_opt_median = np.median(max_time_opt)
max_time_lbd_median = np.median(max_time_lbd)
