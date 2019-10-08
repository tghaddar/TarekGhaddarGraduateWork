import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.close("all")

Q = {}
labels = {}
Q[0] = [0,0]
labels[0] = 4
Q[1] = [-4,-1]
labels[1] = 2
Q[2] = [4,-1]
labels[2] = 2
Q[3] = [-6,-2]
labels[3] = 1
Q[4] = [-2,-2]
labels[4] = 1
Q[5] = [2,-2]
labels[5] = 1
Q[6] = [6,-2]
labels[6] = 1

G = nx.DiGraph()
for i in range(0,7):
  G.add_node(i)

G.add_edge(0,1)
G.add_edge(0,2)
G.add_edge(1,3)
G.add_edge(1,4)
G.add_edge(2,5)
G.add_edge(2,6)

plt.figure()
nx.draw(G,pos=Q,with_labels=False,node_color='r',arrowsize=20)
nx.draw_networkx_labels(G,pos=Q,labels=labels,font_size=16,node_color='r')
plt.savefig("../../figures/binary_tree.pdf")
plt.close()

