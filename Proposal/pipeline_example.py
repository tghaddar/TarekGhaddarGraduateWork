#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 11:52:14 2019

@author: tghaddar
"""

import matplotlib.pyplot as plt

plt.close("all")
def get_ij(ss_id,numrow,numcol):
  j = int(ss_id % numrow)
  i = int((ss_id - j)/numrow)
  return i,j

X = [0,3.33,6.67,10]
Y = [0,3.33,6.67,10]

num_subsets = 9
numrow = 3
numcol = 3

plt.figure()

for n in range(0,num_subsets):
  i,j = get_ij(n,numrow,numcol)
  xmin = X[i]
  xmax = X[i+1]
  ymin = Y[j]
  ymax = Y[j+1]
  
  xvec = [xmin,xmax,xmax,xmin,xmin]
  yvec = [ymin,ymin,ymax,ymax,ymin]
  plt.plot(xvec,yvec,'b')



plt.arrow(0.9,0.9,1.5,1.5,width=0.1,color='r')
plt.arrow(0.9,3.33+0.9,1.5,1.5,width=0.1,color='b')
plt.arrow(0.9+3.33,0.9,1.5,1.5,width=0.1,color='b')
plt.arrow(0.9,0.9+6.67,1.5,1.5,width=0.1,color='g')
plt.arrow(0.9+3.33,3.33+0.9,1.5,1.5,width=0.1,color='g')
plt.arrow(0.9+6.67,0.9,1.5,1.5,width=0.1,color='g')

plt.axis('off')
plt.savefig("../figures/pipeline_example.pdf",bbox_inches='tight')
