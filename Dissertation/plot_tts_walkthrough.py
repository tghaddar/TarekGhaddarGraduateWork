import sys
sys.path.append('/Users/tghaddar/GitHub/TarekGhaddarGraduateWork/sweep_optimizer/3d')
import matplotlib.pyplot as plt
plt.close("all")
#The first portion will plot the quadrant layout.
plt.figure("Quadrant Layout")

x = [-1,1,1,-1,-1]
y = [-1,-1,1,1,-1]
x1 = [0,0]
y1 = [-1, 1]
x2 = [-1,1]
y2 = [0,0]

plt.title("Quadrant Layout")
plt.plot(x,y,'b',x1,y1,'b',x2,y2,'b')
plt.axis('off')

plt.text(-0.5,-0.5,"0",fontweight='bold')
plt.arrow(-0.75,-0.75,0.25,0.25,width=0.1,fill=False,color='r')

plt.text(-0.5,0.45,"1",fontweight='bold')
plt.arrow(-0.75,0.75,0.25,-0.25,width=0.1,fill=False,color='r')

plt.text(0.45,-0.5,"2",fontweight='bold')
plt.arrow(0.75,-0.75,-0.25,0.25,width=0.1,fill=False,color='r')

plt.text(0.5,0.5,"3",fontweight='bold')
plt.arrow(0.75,0.75,-0.25,-0.25,width=0.1,fill=False,color='r')

plt.savefig("figures/quadrant_layout.pdf")
plt.close()
