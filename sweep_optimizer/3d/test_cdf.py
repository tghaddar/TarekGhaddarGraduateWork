import numpy as np
import matplotlib.pyplot as plt
from optimizer import get_column_cdf
plt.close("all")

points = np.genfromtxt("unbalanced_pins_centroid_data")
numcol = 4
gxmin = 0.0
gxmax = 1.0

cdf,bin_edges = get_column_cdf(points,gxmin,gxmax,numcol)
bin_edges = np.delete(bin_edges,0)

grad_cdf = np.diff(cdf)/np.diff(bin_edges)

plt.figure()
plt.plot(cdf)
plt.plot(grad_cdf/max(grad_cdf))
