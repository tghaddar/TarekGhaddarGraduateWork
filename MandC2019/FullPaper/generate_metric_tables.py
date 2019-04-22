import numpy as np

og = np.genfromtxt('opp_side_og_q.csv',delimiter=',')
lbd = np.genfromtxt('opp_side_lbd_q.csv',delimiter=',')

improvement = np.divide(og,lbd)

