import numpy as np

og = np.genfromtxt('opp_side_og_q.csv',delimiter=',')
lbd = np.genfromtxt('opp_side_lbd_q.csv',delimiter=',')

improvement = np.divide(np.subtract(og,lbd),og)

og_tabular = np.savetxt("og_table.csv",og,delimiter=' & ',fmt='%2.3f',newline=' \\\\ \n')
lbd_tabular = np.savetxt("lbd_table.csv",lbd,delimiter=' & ',fmt='%2.3f',newline=' \\\\ \n')
improvement_tabular = np.savetxt("improvement_table.csv",improvement,delimiter=' & ',fmt='%2.3f',newline=' \\\\ \n')
