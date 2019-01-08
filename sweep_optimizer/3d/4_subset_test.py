import networkx as nx
import matplotlib.pyplot as plt
from sweep_solver import add_conflict_weights
from sweep_solver import make_edges_universal

num_nodes = 4
plt.close("all")

G = nx.DiGraph()
G1 = nx.DiGraph()
G2 = nx.DiGraph()
G3 = nx.DiGraph()
for n in range(0,num_nodes):
  G.add_node(n)
  G1.add_node(n)
  G2.add_node(n)
  G3.add_node(n)
  

#Perfectly balanced and orthogonal cut lines.

#Node 0 edges.
G.add_edge(0,1,weight = 1)
G.add_edge(0,2,weight = 1)
G1.add_edge(0,2,weight = 1)
G2.add_edge(0,1,weight = 1)
G3.add_edge(0,-1,weight = 1)

#Node 1 edges.
G.add_edge(1,3,weight = 1)
G1.add_edge(1,0,weight = 1)
G1.add_edge(1,3,weight = 1)
G2.add_edge(1,-1,weight = 1)
G3.add_edge(1,0,weight = 1)

#Node 2 edges.
G.add_edge(2,3,weight = 1)
G1.add_edge(2,-1,weight = 1)
G2.add_edge(2,3,weight = 1)
G2.add_edge(2,0,weight = 1)
G3.add_edge(2,0,weight = 1)

#Node 3 edges.
G.add_edge(3,-1,weight = 1)
G1.add_edge(3,2,weight = 1)
G2.add_edge(3,1,weight = 1)
G3.add_edge(3,2,weight = 1)
G3.add_edge(3,1,weight = 1)

graphs = [G,G1,G2,G3]
graphs = make_edges_universal(graphs)
time_to_solve = [1,1,1,1]



graphs = add_conflict_weights(graphs,time_to_solve)

plt.figure("Graph 0 Final Universal Time")
edge_labels_1 = nx.get_edge_attributes(G,'weight')
nx.draw(G,nx.spectral_layout(G),with_labels = True)
nx.draw_networkx_edge_labels(G,nx.spectral_layout(G),edge_labels=edge_labels_1)

plt.figure("Graph 1 Final Universal Time")
edge_labels_1 = nx.get_edge_attributes(G1,'weight')
nx.draw(G1,nx.spectral_layout(G1),with_labels = True)
nx.draw_networkx_edge_labels(G1,nx.spectral_layout(G1),edge_labels=edge_labels_1)

plt.figure("Graph 2 Final Universal Time")
edge_labels_1 = nx.get_edge_attributes(G2,'weight')
nx.draw(G2,nx.spectral_layout(G2),with_labels = True)
nx.draw_networkx_edge_labels(G2,nx.spectral_layout(G2),edge_labels=edge_labels_1)


plt.figure("Graph 3 Final Universal Time")
edge_labels_1 = nx.get_edge_attributes(G3,'weight')
nx.draw(G3,nx.spectral_layout(G3),with_labels = True)
nx.draw_networkx_edge_labels(G3,nx.spectral_layout(G3),edge_labels=edge_labels_1)