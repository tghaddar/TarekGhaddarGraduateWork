import numpy as np
import matplotlib.pyplot as plt
plt.close("all")
subsets = [2,3,4,5,6,7,8,9,10]
regular_metrics = np.genfromtxt("regular_metrics.txt")
lb_metrics = np.genfromtxt("lb_metrics.txt")
lbd_metrics = np.genfromtxt("lbd_metrics.txt")

plt.figure()
plt.xlabel(r'$\sqrt{\rm{Number\ of\ Subsets}}$')
plt.ylabel('f')
plt.plot(subsets,regular_metrics,'-o',label="No LB")
plt.plot(subsets,lb_metrics,'-o',label="LB")
plt.plot(subsets,lbd_metrics,'-o',label="LBD")
plt.legend(loc='best')
plt.savefig("../../../figures/level2_metric_study.pdf")

f = open("metric_study_table.txt","w")
f.write("\\textbf{$\sqrt{\\text{Num Subsets}}$} & \\bf $f_{\\text{reg}}$ & \\bf $f_{\\text{LB}}$  & \\bf $f_{\\text{LBD}}$\\\ \hline \n")
for i in range(0,len(subsets)):
  f.write( str(subsets[i])+'&'+str(np.round(regular_metrics[i],2))+'&'+str(np.round(lb_metrics[i],2))+'&'+str(np.round(lbd_metrics[i],2)))
  if i < len(subsets)-1:
    f.write("\\\ \hline \n")
  else:
    f.write("\n")

f.close()

#improvement table. 
lb_decrease = np.divide((regular_metrics - lb_metrics),regular_metrics)*100
lbd_decrease = np.divide((regular_metrics - lbd_metrics),regular_metrics)*100
lbd_lb_decrease = np.divide((lb_metrics - lbd_metrics),lb_metrics)*100

f = open("metric_improvement_table.txt",'w')
f.write("\\textbf{$\sqrt{\\text{Num Subsets}}$} & \\bf LB v. Regular  & \\bf LBD v. Regular & \\bf LBD v. LB \\\ \hline \n")
for i in range(0,len(subsets)):
  f.write( str(subsets[i])+'\%&'+str(np.round(lb_decrease[i],2))+'\%&'+str(np.round(lbd_decrease[i],2)) + '\%&'+str(np.round(lbd_lb_decrease[i],2))+'\%')
  if i < len(subsets)-1:
    f.write("\\\ \hline \n")
  else:
    f.write("\n")
f.close()