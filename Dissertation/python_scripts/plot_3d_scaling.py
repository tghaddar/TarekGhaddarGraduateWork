import numpy as np
import matplotlib.pyplot as plt
plt.close("all")

tts_sweep_times = np.genfromtxt("3d_timing_runs.csv",delimiter=',')
perf_model_sweep_times = [0.31,0.418,0.435,0.45,0.46,0.47,0.48]
pdt_sweep_times = [0.34,0.45,0.45,0.50,0.50,0.50,0.53]
cores = [1,8,64,512,1024,	2048,4096]

plt.figure()
plt.title("PDT vs. PDT Perf. Model vs. TTS Estimator")
plt.ylim(0.0,1.0)
plt.xlabel("Cores")
plt.ylabel("Time Per Sweep")
plt.plot(cores,pdt_sweep_times,'--o')
plt.plot(cores,perf_model_sweep_times,'--x')
plt.plot(cores,tts_sweep_times,'--d')
plt.savefig("../../figures/scaling_tts_sweep_times.pdf")


#Scaling Plot
tts_scale = tts_sweep_times[1]/tts_sweep_times
perf_scale = perf_model_sweep_times[1]/np.array(perf_model_sweep_times)
pdt_scale = pdt_sweep_times[1]/np.array(pdt_sweep_times)
plt.figure()
plt.title("Weak Scaling Relative to 8 cores")
plt.ylim(0.0,1.6)
plt.xlabel("Cores")
plt.ylabel("Parallel Efficiency")
plt.plot(cores,pdt_scale,'--o')
plt.plot(cores,perf_scale,'--x')
plt.plot(cores,tts_scale,'--d')
plt.savefig("../../figures/scaling_tts.pdf")