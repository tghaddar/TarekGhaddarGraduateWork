import numpy as np

pdt_data = np.genfromtxt("level2_sweep_data.txt")
pdt_data = np.reshape(pdt_data,(2,9))
pdt_data_median = np.empty(2)
percent_diff = np.empty(2)
for i in range(0,2):
  median = np.median(pdt_data[i])
  pdt_data_median[i] = median