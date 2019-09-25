import numpy as np
import matplotlib.pyplot as plt
plt.close("all")

tts_sweep_times = np.genfromtxt("3d_timing_runs.csv",delimiter=',')
tts_sweep_times = np.array([0.3078936844702819, 0.4121962341283458, 0.43100288184238755, 0.4530049207223008, 0.47202184500297734, 0.4791660081275161, 0.4934647190805943, 0.5290797974992263,0.5449])
perf_model_sweep_times = [0.31,0.42,0.44,0.46,0.49,0.49,0.51,0.55,0.56]
pdt_sweep_times = [0.34,0.45,0.45,0.50,0.50,0.50,0.53,0.60,0.62]
cores = [1,8,64,512,1024,	2048,4096,8192,16384]
stage_count = [128,128,264,540,1068,1084,1116,2172,2236]

#Tick labels for both graphs.
positions = cores
labels = [str(i) for i in cores]

plt.figure()
plt.title("PDT vs. PDT Perf. Model vs. TTS Estimator")
plt.ylim(0.0,1.0)
plt.xlabel("Cores")
plt.ylabel("Time Per Sweep (s)")
plt.semilogx(cores,pdt_sweep_times,'--o',label='PDT')
plt.semilogx(cores,perf_model_sweep_times,'--x',label='Perf. Model')
plt.semilogx(cores,tts_sweep_times,'--d',label='TTS Estimator')
plt.legend(loc='best')
plt.savefig("../../figures/scaling_tts_sweep_times.pdf")


#Scaling Plot
tts_scale = tts_sweep_times[1]/tts_sweep_times
perf_scale = perf_model_sweep_times[1]/np.array(perf_model_sweep_times)
pdt_scale = pdt_sweep_times[1]/np.array(pdt_sweep_times)
plt.figure()
plt.xticks(positions,labels)
plt.title("Weak Scaling Relative to 8 cores")
plt.ylim(0.0,1.6)
plt.xlabel("Cores")
plt.ylabel("Parallel Efficiency")
plt.semilogx(cores,pdt_scale,'--o',label='PDT')
plt.semilogx(cores,perf_scale,'--x',label='Perf. Model')
plt.semilogx(cores,tts_scale,'--d',label='TTS Estimator')
plt.legend(loc='best')
plt.savefig("../../figures/scaling_tts.pdf")

plt.figure()
plt.title("Stage Counts for TTS and Perf. Model")
plt.xlabel("Cores")
plt.ylabel("Stage Count")
plt.semilogx(cores,stage_count,'-b',label='Perf. Model')
plt.semilogx(cores,stage_count,'ro',label='TTS')
plt.legend(loc='best')
plt.savefig("../../figures/scaling_stagecount.pdf")

percent_diff_tts_pdt = [None]*len(cores)
percent_diff_perf_model_pdt = [None]*len(cores)
percent_diff_tts_perf_model = [None]*len(cores)

for i in range(0,len(cores)):
  percent_diff_tts_pdt[i] = abs(tts_sweep_times[i] - pdt_sweep_times[i])/pdt_sweep_times[i]*100
  percent_diff_perf_model_pdt[i] = abs(perf_model_sweep_times[i] - pdt_sweep_times[i])/pdt_sweep_times[i]*100
  percent_diff_tts_perf_model[i] = abs(perf_model_sweep_times[i] - tts_sweep_times[i])/perf_model_sweep_times[i]*100
  
f = open("percent_diffs.txt",'w')
f.write("Cores & PDT v. Perf. & Perf. v. TTS & PDT v. TTS \\\ \hline"+'\n')
for i in range(0,len(cores)):
  f.write(str(cores[i])+"\%&"+str(np.round(percent_diff_perf_model_pdt[i],2))+"\%&"+str(np.round(percent_diff_tts_perf_model[i],2))+"\%&"+str(np.round(percent_diff_tts_pdt[i],2))+"\%")
  if i < len(cores)-1:
    f.write(" \\\ \hline" +'\n')
  else:
    f.write('\n')

f.close()