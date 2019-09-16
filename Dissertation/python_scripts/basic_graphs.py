import networkx as nx
import matplotlib.pyplot as plt
plt.close("all")

G = nx.Graph()
G.add_nodes_from([0,1,2,3])
G.add_edge(0,1)
G.add_edge(0,2)
G.add_edge(2,3)
G.add_edge(1,3)
Q = {}
Q[0] = [0,0]
Q[1] = [1,1]
Q[2] = [1,-1]
Q[3] = [2,0]

plt.figure()
nx.draw(G,Q,with_labels=True,node_size=2000,font_size=20)
plt.savefig("../../figures/undirected_graph.pdf")

G = nx.DiGraph()
G.add_nodes_from([0,1,2,3])
G.add_edge(0,1)
G.add_edge(0,2)
G.add_edge(2,3)
G.add_edge(1,3)
Q = {}
Q[0] = [0,0]
Q[1] = [1,1]
Q[2] = [1,-1]
Q[3] = [2,0]
plt.figure()
nx.draw(G,Q,with_labels=True,arrowsize=20,node_size=2000,font_size=20)
plt.savefig("../../figures/directed_graph.pdf")

G = nx.DiGraph()
G.add_nodes_from([0,1,2,3])
G.add_edge(0,1)
G.add_edge(0,2)
G.add_edge(2,3)
G.add_edge(1,3)
G.add_edge(3,1)
Q = {}
Q[0] = [0,0]
Q[1] = [1,1]
Q[2] = [1,-1]
Q[3] = [2,0]
plt.figure()
nx.draw(G,Q,with_labels=True,arrowsize=20,node_size=2000,font_size=20)
plt.savefig("../../figures/cycle_example.pdf")

G = nx.DiGraph()
G.add_nodes_from([0,1,2,3])
G.add_edge(0,1,weight=1.0)
G.add_edge(0,2,weight=1.0)
G.add_edge(2,3,weight=1.0)
G.add_edge(1,3,weight=3.0)
Q = {}
Q[0] = [0,0]
Q[1] = [1,1]
Q[2] = [1,-1]
Q[3] = [2,0]
plt.figure()
edge_labels = (nx.get_edge_attributes(G,'weight'))
nx.draw(G,Q,with_labels=True,arrowsize=20,node_size=2000,font_size=20)
nx.draw_networkx_edge_labels(G,Q,edge_labels=edge_labels,font_size=14)
plt.savefig("../../figures/weighted_directed_graph.pdf")

G = nx.DiGraph()
G.add_nodes_from([0,1,2,3])
G.add_edge(0,1,weight=-1.0)
G.add_edge(0,2,weight=-1.0)
G.add_edge(2,3,weight=-1.0)
G.add_edge(1,3,weight=-3.0)
Q = {}
Q[0] = [0,0]
Q[1] = [1,1]
Q[2] = [1,-1]
Q[3] = [2,0]
plt.figure()
edge_labels = (nx.get_edge_attributes(G,'weight'))
nx.draw(G,Q,with_labels=True,arrowsize=20,node_size=2000,font_size=20)
nx.draw_networkx_edge_labels(G,Q,edge_labels=edge_labels,font_size=20)
plt.savefig("../../figures/negative_weighted_directed_graph.pdf")