import matplotlib.pyplot as plt
#Storing data for runs. 

plt.close("all")
nss = [2,3,4,5,6,7,8,9,10]
nss_lb = [2,3,4,5,6,7,9]
f_reg = [1.38462,2.15702,3.69231,4.28571,5.72143,6.77174,12.9231,9.072,10.0264]
f_lb = [1.384615,2.04545,3.1453,3.74564,4.42105,6.37778,9.53086]
f_lbd = [1.07692,1.13445,1.11498,1.11386,1.30796,1.33636,1.50276,1.7234,1.51188]

#Only to 7
nss = [2,3,4,5,6,7]
nss_lb = [2,3,4,5,6,7]


f_reg = [1.38462,2.15702,3.69231,4.28571,5.72143,6.77174]
f_lb = [1.384615,2.04545,3.1453,3.74564,4.42105,6.37778]
f_lbd = [1.07692,1.13445,1.11498,1.11386,1.30796,1.33636]

plt.figure()
plt.xlabel('Subsets in x and y')
plt.ylabel('f')
plt.plot(nss,f_reg,'o',label='Regular')
plt.plot(nss_lb,f_lb,'o',label='LB')
plt.plot(nss,f_lbd,'o',label='LBD')
plt.legend(loc='best')
plt.savefig("../../figures/lbd_results.pdf")

f_reg_sparse = [1.7619,3.41379,5.82759,9.7619,10.022,11.1726,12.4389,8.81481,9.7619]
f_lb_sparse = [1.7619,2.90204,4.29572,5]
f_lbd_sparse = [1.10638,1.26512,1.28467,1.4095,1.26606,7,8,1.22727,1.38067]