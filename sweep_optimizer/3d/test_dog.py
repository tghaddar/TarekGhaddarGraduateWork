import networkx as nx
import matplotlib.pyplot as plt
import warnings
from sweep_solver import get_DOG
from sweep_solver import get_heaviest_path
warnings.filterwarnings("ignore", category=DeprecationWarning)

plt.close("all")

G = nx.DiGraph()
G2 = nx.DiGraph()

num_nodes = 5

for n in range(0,num_nodes):
  G.add_node(n)
  G2.add_node(n)

G.add_edge(0,1,weight = 3)
G.add_edge(1,2,weight = 10)
G.add_edge(2,3,weight = 2)
G.add_edge(3,4,weight = 8)

G2.add_edge(4,3,weight = 1)
G2.add_edge(3,2,weight = 8)
G2.add_edge(2,1,weight = 2)
G2.add_edge(1,0,weight = 10)


path1 = nx.all_simple_paths(G,0,4)
path2 = nx.all_simple_paths(G2,4,0)

graphs = [G,G2]
paths = [path1,path2]

#Testing the DOG remaining method.
path, path_sum = get_heaviest_path(G,path1)
dog = get_DOG(G,path,2)