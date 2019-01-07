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
  

