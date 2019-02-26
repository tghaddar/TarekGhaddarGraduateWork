# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 12:27:10 2019

@author: tghad
"""

import matplotlib.pyplot as plt
plt.close("all")
plt.figure()
y = [36.56/1000, 119.97/1000, 470.61/1000, 2.2, 9.95, 11.22,2.76*60]
x = [10,12,14,16,18,20,22]
plt.plot(x,y)