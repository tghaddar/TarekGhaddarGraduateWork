import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import numpy as np
import mpmath

warnings.filterwarnings("ignore", category=DeprecationWarning)

max_times = np.genfromtxt("max_times.csv")
plt.close("all")
#The mesh density function.
f = lambda x,y: x+y

mpmath.splot(f,[0,10],[0,10])



x = max_times[:,0]
y1 = max_times[:,1]
y2 = max_times[:,2]
time = max_times[:,3]
mintime = min(time)


fig = plt.figure("Result")
ax = fig.gca(projection='3d')

img = ax.scatter(y1,y2,x,c=time,cmap = plt.cool())
fig.colorbar(img)
