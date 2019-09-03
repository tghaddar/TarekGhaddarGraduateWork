import numpy as np
from optimizer import get_column_cdf
import matplotlib.pyplot as plt
plt.close("all")

vert_data = np.genfromtxt("sparse_pins_vert_data",delimiter = ' ')

cdf,bin_edges = get_column_cdf(vert_data,0,10,5)

grad_cdf = np.diff(cdf)/np.diff(bin_edges)
grad_cdf = np.insert(grad_cdf,0,0.0)

plt.figure()
plt.plot(bin_edges,cdf)
plt.plot(bin_edges,grad_cdf/max(grad_cdf))