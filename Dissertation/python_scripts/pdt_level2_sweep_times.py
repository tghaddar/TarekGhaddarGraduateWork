import numpy as np
import matplotlib.pyplot as plt
plt.close("all")

max_time_reg = 0.06480723685287824
max_time_lb = 0.05346907738394505
pdt_data = np.genfromtxt("level2_sweep_data.txt")
pdt_data = np.reshape(pdt_data,(2,10))
tts_data = [max_time_reg,max_time_lb]
pdt_data_median = np.empty(2)
pdt_data_min = np.empty(2)
pdt_data_max = np.empty(2)
percent_diff = np.empty(2)
for i in range(0,2):
  tts = tts_data[i]
  median = np.median(pdt_data[i])
  pdt_data_median[i] = median
  pdt_data_min[i] = np.min(pdt_data[i])
  pdt_data_max[i] = np.max(pdt_data[i])
  percent_diff[i] = abs(tts - median)/median*100
  



f = open("level2_percent_diff.txt",'w')
f.write("\\textbf{$\sqrt{\\text{Case}}$} & \\bf PDT vs. TTS \\\ \hline \n")
f.write("Regular & ")
f.write(str(np.round(percent_diff[0],2)) + "\%")
f.write(" \\\ \hline")
f.write("Balanced & ")
f.write(str(np.round(percent_diff[1],2)) + "\%")
