#Plots the 2D verification.
import numpy as np
import matplotlib.pyplot as plt
plt.close("all")

computation_time = np.genfromtxt("2d_verification_results.csv")
computation_time_mild_random = np.genfromtxt("2d_verification_mild_random_results.csv")
computation_time_random = np.genfromtxt("2d_verification_random_results.csv")
computation_time_worst = np.genfromtxt("2d_verification_worst_results.csv")
angle_range,spatial_range = np.shape(computation_time)

num_angles = list(range(1,angle_range+1))
num_subsets = list(range(2,spatial_range+2))

plt.figure("2D Verification")
plt.title("2D Verification")
plt.xlabel("Number of Subsets in X and Y")
plt.ylabel("Number of Stages")

for angles in num_angles:
  angle = str(int(angles))
  computation_time_this_angle = computation_time[angles-1]
  analytical = [(4*(angles-1) + 2*s + 2*(s%2)) for s in num_subsets]
  plt.plot(num_subsets,computation_time_this_angle,label=angle + " angles")
  plt.plot(num_subsets,analytical,'o')

plt.legend(loc='best')
plt.savefig("regular_verification.pdf")


mild_random_results = np.empty([angle_range,spatial_range])
f = open('results_mild_random.txt')
mild_random_results_matlab = f.readlines()
f.close()
counter = 0
for s in range(0,spatial_range):
  for a in range(0,angle_range):
    mild_random_results[a][s] = int(mild_random_results_matlab[counter])
    counter += 1


plt.figure("2D Verification Mild Random")
plt.title("2D Verification Mild Random")
plt.xlabel("Number of Subsets in X and Y")
plt.ylabel("Number of Stages")
for angles in num_angles:
  angle = str(int(angles))
  computation_time_this_angle = computation_time_mild_random[angles-1]
  mild_random_results_this_angle = mild_random_results[angles-1]
  plt.plot(num_subsets,computation_time_this_angle,label=angle + " angles")
  plt.plot(num_subsets,mild_random_results_this_angle,'o')

plt.legend(loc='best')
plt.savefig("mild_random_verification.pdf")



random_results = np.empty([angle_range,spatial_range])
f = open('results_random.txt')
random_results_matlab = f.readlines()
f.close()
counter = 0
for s in range(0,spatial_range):
  for a in range(0,angle_range):
    random_results[a][s] = int(random_results_matlab[counter])
    counter += 1
plt.figure("2D Verification Random")
plt.title("2D Verification Random")
plt.xlabel("Number of Subsets in X and Y")
plt.ylabel("Number of Stages")
for angles in num_angles:
  angle = str(int(angles))
  computation_time_this_angle = computation_time_random[angles-1]
  random_results_this_angle = random_results[angles-1]
  plt.plot(num_subsets,computation_time_this_angle,label=angle + " angles")
  plt.plot(num_subsets,random_results_this_angle,'o')

plt.legend(loc='best')
plt.savefig("random_verification.pdf")


worst_results = np.empty([angle_range,spatial_range])
f = open('results_worst.txt')
worst_results_matlab = f.readlines()
f.close()
counter = 0
for s in range(0,spatial_range):
  for a in range(0,angle_range):
    worst_results[a][s] = int(worst_results_matlab[counter])
    counter += 1

plt.figure("2D Verification Worst")
plt.title("2D Verification Worst")
plt.xlabel("Number of Subsets in X and Y")
plt.ylabel("Number of Stages")
for angles in num_angles:
  angle = str(int(angles))
  computation_time_this_angle = computation_time_worst[angles-1]
  worst_results_this_angle = worst_results[angles-1]
  plt.plot(num_subsets,computation_time_this_angle,label=angle + " angles")
  plt.plot(num_subsets,worst_results_this_angle,'o')

plt.legend(loc='best')
plt.savefig("worst_verification.pdf")