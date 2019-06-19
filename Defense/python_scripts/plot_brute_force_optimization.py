import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.close("all")


max_times = np.genfromtxt("max_times.csv")
x = max_times[:,0]
y0 = max_times[:,1]
y1 = max_times[:,2]
time = max_times[:,3]

fig = plt.figure("Brute Force Result")
ax = fig.gca(projection='3d')
img=ax.scatter(x,y0,y1,c=time,cmap=plt.cool())
ax.set_xlabel("x")
ax.set_ylabel("y0")
ax.set_zlabel("y1")
fig.colorbar(img)
plt.savefig("../../figures/brute_force_result.pdf")
plt.close()