import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.close("all")


max_times = np.genfromtxt("max_times_0.1.csv")
x = max_times[:,0]
y0 = max_times[:,1]
y1 = max_times[:,2]
time = max_times[:,3]

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
yeet=ax.scatter(x,y0,y1,c=time,cmap=plt.hot())
fig.colorbar(yeet)
plt.savefig("../../figures/brute_force_result.pdf")
plt.close()