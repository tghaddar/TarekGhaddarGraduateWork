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
  
tts_data = np.genfromtxt("tts_level2_data")


plt.figure()
plt.plot([1,2,3,4,5],pdt_data_median,'-o',label="PDT")
plt.plot([1,2,3,4,5],tts_data,'-o',label="TTS")
plt.legend(loc='best')
plt.ylabel("Sweep Time (s)")
plt.xticks([1,2,3,4,5],("Regular", "Balanced", "LB", "LBD", "Opt"))
plt.savefig("../../figures/level2_sweep_comp.pdf")