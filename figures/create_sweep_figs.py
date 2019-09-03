import matplotlib.pyplot as plt
plt.close("all")

plt.figure()
plt.xlim(-1,4)
plt.ylim(-1,4)
plt.axis("off")

#Big box.
x = [0,4,4,0,0]
y = [0,0,4,4,0]
plt.plot(x,y,'k')
#Horizontal lines.
x = [0,4]
plt.plot(x,[1,1],'k')
plt.plot(x,[2,2],'k')
plt.plot(x,[3,3],'k')
#Vertical lines.
y = [0,4]
plt.plot([1,1],y,'k')
plt.plot([2,2],y,'k')
plt.plot([3,3],y,'k')
#Text for stage 1.
x = 0.4
y = 0.4
plt.text(x,y,'1',fontsize=20,fontweight='bold')

#Stage 2.
x = 0.4
y = 1.4
plt.text(x,y,'2',fontsize=20,fontweight='bold',color='r')
x = 1.4
y = 0.4
plt.text(x,y,'2',fontsize=20,fontweight='bold',color='r')

#Stage 3.
x = 0.4
y = 2.4
plt.text(x,y,'3',fontsize=20,fontweight='bold',color='b')
x = 1.4
y = 1.4
plt.text(x,y,'3',fontsize=20,fontweight='bold',color='b')
x = 2.4
y = 0.4
plt.text(x,y,'3',fontsize=20,fontweight='bold',color='b')

#Stage 4.
x = 0.4
y = 3.4
plt.text(x,y,'4',fontsize=20,fontweight='bold',color='g')
x = 1.4
y = 2.4
plt.text(x,y,'4',fontsize=20,fontweight='bold',color='g')
x = 2.4
y = 1.4
plt.text(x,y,'4',fontsize=20,fontweight='bold',color='g')
x = 3.4
y = 0.4
plt.text(x,y,'4',fontsize=20,fontweight='bold',color='g')

#Stage 5.
x = 1.4
y = 3.4
plt.text(x,y,'5',fontsize=20,fontweight='bold',color='m')
x = 2.4
y = 2.4
plt.text(x,y,'5',fontsize=20,fontweight='bold',color='m')
x = 3.4
y = 1.4
plt.text(x,y,'5',fontsize=20,fontweight='bold',color='m')

#Stage 6.
x = 2.4
y = 3.4
plt.text(x,y,'6',fontsize=20,fontweight='bold',color='c')
x = 3.4
y = 2.4
plt.text(x,y,'6',fontsize=20,fontweight='bold',color='c')

#Stage 7.
x = 3.4
y = 3.4
plt.text(x,y,'7',fontsize=20,fontweight='bold',color='k')

#Arrow and omega.
plt.arrow(-0.5,-0.5,0.3,0.3,width=0.05,color='k')
plt.text(-0.75,-0.75,r'$\Omega$',fontsize=20,fontweight='extra bold')
plt.savefig("StructuredMesh.pdf")


#Unstructured Mesh.
plt.figure()
x = [0,1,1.5,1,0,-0.5,0]
y = [0,0,1,2,2,1,0]
plt.plot(x,y,'k')
x = [0,0.25]
y = [0,0.5]
plt.plot(x,y,'k')
x = [-0.5,0.25]
y = [1,0.5]
plt.plot(x,y,'k')
x = [0.25,0.5]
y = [0.5,1]
plt.plot(x,y,'k')
x = [-0.5,0.5]
y = [1,1]
plt.plot(x,y,'k')
x = [0.25,0.95]
y = [0.5,0.75]
plt.plot(x,y,'k')
x = [0.5,0.95]
y = [1,0.75]
plt.plot(x,y,'k')
x = [0,0.5]
y = [2,1]
plt.plot(x,y,'k')
x = [0.6,0.5]
y = [2,1]
plt.plot(x,y,'k')
x = [0.25,1]
y = [0.5,0]
plt.plot(x,y,'k')
x = [0.95,1]
y = [0.75,0]
plt.plot(x,y,'k')
x = [0.95,1.5]
y = [0.75,1]
plt.plot(x,y,'k')
x = [0.95,0.9]
y = [0.75,1.5]
plt.plot(x,y,'k')
x = [0.5,0.9]
y = [1,1.5]
plt.plot(x,y,'k')
x = [0.9,0.6]
y = [1.5,2]
plt.plot(x,y,'k')
x = [0.9,1]
y = [1.5,2]
plt.plot(x,y,'k')
x = [0.9,1.5]
y = [1.5,1]
plt.plot(x,y,'k')


