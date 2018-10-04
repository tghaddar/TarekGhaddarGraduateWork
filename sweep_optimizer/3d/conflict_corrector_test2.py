#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 13:40:43 2018
A different method of correcting conflicts with 4 paths.
@author: tghaddar
"""

import networkx as nx

G = nx.DiGraph()
G1 = nx.DiGraph()
G2 = nx.DiGraph()
G3 = nx.DiGraph()

for n in range(0,4):
  G.add_node(n)
  G1.add_node(n)
  G2.add_node(n)
  G3.add_node(n)

G.add_edge(0,1)
G.add_edge(0,2)
G.add_edge(1,2)
G.add_edge(1,3)
G.add_edge(2,3)

G1.add_edge(1,0)
G1.add_edge(1,2)
G1.add_edge(1,3)