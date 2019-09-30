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

plt.text(1.5,1,"Angle 3")
plt.text(1.5+3.33,1,"Angle 2")
plt.text(1.5+6.67,1,"Angle 1")
plt.text(1.5,1+3.33,"Angle 2")
plt.text(1.5,1+6.67,"Angle 1")
plt.text(1.5+3.33,1+3.33,"Angle 1")

plt.axis('off')
plt.savefig("../figures/pipeline_example.pdf",bbox_inches='tight')

plt.figure()

numrow = 2
numcol = 2

plt.plot([0,0,5,5,0],[0,10,10,0,0],'-b')
plt.plot([5,5,10,10,5],[0,10,10,0,0],'-b')
plt.plot([0,10],[2,2],'--k')
plt.plot([0,10],[4,4],'--k')
plt.plot([0,10],[6,6],'--k')
plt.plot([0,10],[8,8],'--k')

plt.text(2.0,1.0,"Angle 3")
plt.text(2.0,3.0,"Angle 2")
plt.text(2.0,5.0,"Angle 1")
plt.text(7.0,1.0,"Angle 2")
plt.text(7.0,3.0,"Angle 1")

plt.xlabel("X")
plt.ylabel("Z")
plt.axis("off")
plt.savefig("../../figures/pipeline_example_3d.pdf",bbox_inches='tight')
