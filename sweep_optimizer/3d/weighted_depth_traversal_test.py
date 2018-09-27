#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 12:03:31 2018
#Unit test for the graph depth traversal.
@author: tghaddar
"""

import networkx as nx

G = nx.DiGraph()

for n in range(0,8):
  G.add_node(n)

