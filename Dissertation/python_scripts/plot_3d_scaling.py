import numpy as np
import matplotlib.pyplot as plt
plt.close("all")

tts_sweep_times = np.genfromtxt("3d_timing_runs.csv",delimiter=',')
tts_sweep_times = np.array([0.3078936844702819, 0.4121962341283458, 0.43100288184238755, 0.4530049207223008, 0.47202184500297734, 0.4791660081275161, 0.4934647190805943, 0.5290797974992263])
perf_model_sweep_times = [0.31,0.42,0.44,0.46,0.49,0.49,0.51,0.55]
pdt_sweep_times = [0.34,0.45,0.45,0.50,0.50,0.50,0.53,0.60]
cores = [1,8,64,512,1024,	2048,4096,8192]

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