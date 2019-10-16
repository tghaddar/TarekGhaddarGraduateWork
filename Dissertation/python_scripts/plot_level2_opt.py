import numpy as np
import matplotlib.pyplot as plt
plt.close("all")

pdt_data = np.genfromtxt("level2_opt_sweep_data.txt")
pdt_data = np.reshape(pdt_data,(5,10))
pdt_data_median = np.empty(5)
pdt_data_min = np.empty(5)
pdt_data_max = np.empty(5)
for i in range(0,5):
  median = np.median(pdt_data[i])
  pdt_data_median[i] = median
  pdt_data_min[i] = np.min(pdt_data[i])
  pdt_data_max[i] = np.max(pdt_data[i])
  
tts_data = np.genfromtxt("tts_level2_data_best")
pdt_data_median[4] = 0.051


plt.figure()
plt.grid(True,axis='y')
plt.plot([1,2,3,4,5],pdt_data_median,'-o',label="PDT")
plt.plot([1,2,3,4,5],tts_data,'-o',label="TTS")
plt.legend(loc='best')
plt.ylabel("TTS (s)")
plt.xticks([1,2,3,4,5],("Regular", "Balanced", "LB", "LBD", "Bin."))
plt.xlabel("Partition Type")
plt.savefig("../../figures/level2_sweep_comp_best.pdf")