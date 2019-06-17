import matplotlib.pyplot as plt
import numpy as np
plt.close("all")


subsets_lbd = [2,3,4,6]
subsets_lb = [2,3,4,5,6,7,8,9,10]

ratio_pdt_lb = np.genfromtxt("ratio_pdt_lb")
ratio_tts_lb = np.genfromtxt("ratio_tts_lb")
ratio_pdt_lbd = np.genfromtxt("ratio_pdt_sparse_lbd")
ratio_tts_lbd = np.genfromtxt("ratio_tts_sparse_lbd")
#ratio_pdt_lbd = np.delete(ratio_pdt_lbd,5)
#ratio_tts_lbd = np.delete(ratio_tts_lbd,5)

#diff_lb = abs(ratio_pdt_lb-ratio_tts_lb)
#percent_diff_lb = diff_lb/ratio_pdt_lb
diff_lbd = abs(ratio_pdt_lbd-ratio_tts_lbd)
percent_diff_lbd = (diff_lbd/ratio_pdt_lbd)*100

#plt.figure("LB")
#plt.title("Ratios for PDT and TTS for Load Balanced Cuts")
#plt.xlabel("Number of Subsets")
#plt.ylabel("Ratios")
#plt.plot(subsets_lb,ratio_tts_lb,'-b',label='TTS')
#plt.plot(subsets_lb,ratio_pdt_lb,'or',label='PDT')
#plt.legend(loc='best')
##
#plt.figure("LB Percent Diff")
#plt.plot(subsets_lb,percent_diff_lb)

plt.figure("LBD")
plt.title("Ratios for PDT and TTS for Load Balanced by Dimension Cuts")
plt.xlabel("Number of Subsets")
plt.ylabel("Ratios")
plt.plot(subsets_lbd,ratio_tts_lbd,'b',label='TTS')
plt.plot(subsets_lbd,ratio_pdt_lbd,'or',label='PDT')
plt.legend(loc='best')

plt.figure("LBD Percent Difference")
plt.title("Percent Difference between PDT Ratio and TTS Ratio for LBD")
plt.xlabel("Number of Subsets in each Dimension")
plt.ylabel("Percent Difference")
plt.plot(subsets_lbd,percent_diff_lbd,'-o')