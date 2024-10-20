import pandas as pd
import numpy as np
import scipy
import networkx as nx
import time
from rdkit import Chem
import dgl
import matplotlib.pyplot as plt

from multi-persistence_diagram_betti0_extraction import *


name = '2D'

with open('2D_dataupdiso.csv', 'r') as f:
        data = f.readlines()

num_graph = len(data)

Graphs_info = {}
for i in range(num_graph):
    print(i)
    string = data[i]
    smiles = string.strip().split(',')
    print(smiles)
    try:
        Graphs_info[i] = get_info(smiles[0])
    except:
        print(i)
        continue

                              
threshold_array_bond = get_thresholds_bond(num_graph, Graphs_info) # thresholds for degree function

threshold_array_weight = get_thresholds_weight(num_graph, Graphs_info)   # thresholds for degree function

features_degree_sub = np.array(degree_sub(num_graph, Graphs_info, threshold_array_bond,threshold_array_weight))

bet0_degree_sub = pd.DataFrame(features_degree_sub)
bet0_degree_sub.to_csv(name + "bet0iso.csv")
