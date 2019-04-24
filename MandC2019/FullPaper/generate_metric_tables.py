import numpy as np

og = np.genfromtxt('opp_side_og_q.csv',delimiter=',')
lbd = np.genfromtxt('opp_side_lbd_q.csv',delimiter=',')
no_lb = np.genfromtxt('opp_side_no_lb_q.csv',delimiter=',')

og_improvement = np.abs(np.divide(np.subtract(no_lb,og),no_lb))
lbd_improvement = np.abs(np.divide(np.subtract(no_lb,lbd),no_lb))

lb_improvement = np.divide(np.subtract(og,lbd),og)


og_tabular = np.savetxt("og_table.csv",og,delimiter=' & ',fmt='%2.3f',newline=' \\\\ \hline \n')
lbd_tabular = np.savetxt("lbd_table.csv",lbd,delimiter=' & ',fmt='%2.3f',newline=' \\\\ \hline \n')

og_improvement_tabular = np.savetxt("og_improvement_table.csv",og_improvement,delimiter=' & ',fmt='%2.3f',newline=' \\\\ \hline \n')
lbd_improvement_tabular = np.savetxt("lbd_improvement_table.csv",lbd_improvement,delimiter=' & ',fmt='%2.3f',newline=' \\\\ \hline \n')


lb_improvement_tabular = np.savetxt("lb_improvement_table.csv",lb_improvement,delimiter=' & ',fmt='%2.3f',newline=' \\\\ \hline \n')
