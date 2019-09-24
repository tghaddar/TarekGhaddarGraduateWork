import numpy as np
import matplotlib.pyplot as plt
plt.close("all")

tts_sweep_times = np.genfromtxt("3d_timing_runs.csv",delimiter=',')
tts_sweep_times = np.array([0.3078936844702819, 0.4121962341283458, 0.43100288184238755, 0.4530049207223008, 0.47202184500297734, 0.4791660081275161, 0.4934647190805943, 0.5290797974992263,0.5449])
perf_model_sweep_times = [0.31,0.42,0.44,0.46,0.49,0.49,0.51,0.55,0.56]
pdt_sweep_times = [0.34,0.45,0.45,0.50,0.50,0.50,0.53,0.60,0.62]
cores = [1,8,64,512,1024,	2048,4096,8192,16384]

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