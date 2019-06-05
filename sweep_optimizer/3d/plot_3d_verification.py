import numpy as np
import matplotlib.pyplot as plt
plt.close("all")



computation_time = np.genfromtxt("3d_verification_results.csv")
angle_range,spatial_range = np.shape(computation_time)
num_angles = list(range(1,angle_range+1))
num_subsets = list(range(2,spatial_range+2))


plt.figure("3D Verification")
plt.title("3D Verification")
plt.xlabel("Number of Subsets in X,Y, and Z")
plt.ylabel("Number of Stages")






for angles in num_angles:
  angle = str(int(angles))
  tasks_per_processor = angles*8
  computation_time_this_angle = computation_time[angles-1]
  analytical = [(tasks_per_processor + 3*(s - (2 - s%2))) for s in num_subsets]
  plt.plot(num_subsets,computation_time_this_angle,label=angle + " angles")
  plt.plot(num_subsets,analytical,'o')

plt.legend(loc='best')
plt.savefig("3d_verification.pdf")
