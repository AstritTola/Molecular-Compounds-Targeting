import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import pyflagser
import networkx as nx


def get_thresholds_bond(num_graph, input_info):
    
    thresh_list = list()

    for graph_id in range(num_graph):
        print(graph_id)
        for v in list(input_info[graph_id][1].keys()):
            thresh_list.append(input_info[graph_id][1][v]['bond_type'])
    
    I1 = np.unique(thresh_list)

    return I1


def get_thresholds_weight(num_graph, input_info):
    
    thresh_list = list()

    for graph_id in range(num_graph):
        print(graph_id)
        for v in list(input_info[graph_id][0].keys()):
            thresh_list.append(input_info[graph_id][0][v]['atomic_num'])
    
    I1 = np.unique(thresh_list)

    return I1




def degree_sub(num_graph, input_info, threshold_array, thresh):
    threshold_array = sorted(threshold_array)  # thresholds for atomic weight
    thresh = sorted(thresh) # thresholds for bond type
    Bet0 = list()
  
    for graph_id in range(num_graph):
       
        B0=list()
        wgt = list()
        bnd = []
        
        
        graph_edges = list(input_info[graph_id][1].keys())
        
        graph = nx.from_edgelist(graph_edges)
        
        
        if len(threshold_array) != 0:
            
            X = len(list(graph.nodes()))
            
            for v in range(X):

                wgt.append((v,input_info[graph_id][0][v]['atomic_num']))
                
            for w in list(input_info[graph_id][1].keys()):
                
                bnd.append((w,input_info[graph_id][1][w]['bond_type']))

        Ed = {}
        for degr1 in thresh:
                Gindex = [index for (index, weight) in wgt if weight <= degr1]
                Ed[degr1] = Gindex
        
        
        for degr in threshold_array:
            
            b0=[]
            
            Rindex = [index for (index, bond) in bnd if bond <= degr]
            
            
            for degr1 in thresh:
                ed = np.intersect1d(Rindex,Ed[degr1])
                sub = graph.subgraph(ed)

            
                if len(ed)>0:
                    adjacency_matrix = nx.adjacency_matrix(sub)
                    adjacency_matrix = adjacency_matrix.todense() 

                    diagr = pyflagser.flagser_unweighted(adjacency_matrix, min_dimension=0, max_dimension=1, directed=False, coeff=2, approximation=None) # applying clique-complexes

                    b0.append(diagr['betti'][0])
                else:

                    b0.append(0)                    
                    
            B0.append(b0)

        Bet0.append(np.array(B0).flatten()) # convert 2D-vectors into 1D-vectors        

    return Bet0, Bet1
