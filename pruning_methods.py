# The following code calculates the cosine similarity weight importance index for each filter in an unpruned CNN model.
# The pruning strategy is based on filtering the lowest weighted filters based on the index.
# It includes the following steps:
# 1- Importing required packages
# 2- Loading the original unpruned model
# 3- Defining a rank1_approximation function
# 4- Implementing CS_WASPAA function to calculate the cosine similarity weight importance index
# 5- Applying the function to each layer in the model
# 6- Saving the resulting index in a file named "sim_index<j>.npy" where j is the layer number.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np    
from time import process_time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from datetime import datetime
import numpy as np
import networkx as nx
import matplotlib as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# use seaborn plotting defaults
import seaborn as sns; sns.set()

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_circles
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


#%%

default_params = {
    'node_color': 'blue',
    'node_size': 20,
    'node_shape': 'o',
    'edge_color': 'gray',
    'width': 0.1,
    'style': 'solid',
    'with_labels': False,
    'font_size': 10,
}
def draw_networkx_without_self_edges(G, pos=None, **kwargs):
    kwargs['edgelist'] = [(u, v) for (u, v) in G.edges() if u != v]
    nx.draw_networkx(G, pos=pos, **kwargs)

def greedy_filter_selection(G, T):
    H = set()
    for x in G.nodes():
        if x not in H:
            Y = set([x] + list(G.neighbors(x)))
            h, max_degree, max_weights = x, G.degree(x), G.nodes[x]['weight']
            for y in Y:
                if G.degree(y) > max_degree or (G.degree(y) == max_degree and G.nodes[y]['weight'] > max_weights):
                    h, max_degree, max_weights = y, G.degree(y), G.nodes[y]['weight']
            H.add(h)
    H.update(T)
    return H
# define a rank1 approximation function
def rank1_apporx(data):
    u,w,v= np.linalg.svd(data)
    M = np.matmul(np.reshape(u[:,0],(-1,1)),np.reshape(v[0,:],(1,-1)))
    M_prototype = M[:,0]/np.linalg.norm(M[:,0],2)
    return M_prototype

def filter_pruning_using_ranked_betweenness(filters, num_filters_to_prune, ascending=False):
    num_filters, channel_size = filters.shape
    cosine_similarities = cosine_similarity(filters)
    
    # Create a graph
    G = nx.Graph()
    G.add_nodes_from(range(num_filters))
    
    for i in range(num_filters):
        for j in range(i+1, num_filters):
            G.add_edge(i, j, weight=cosine_similarities[i, j])
    
    # Compute betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
    
    # Rank the nodes
    ranked_nodes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=not ascending)
    
    # Identify filters to prune
    filters_to_prune = [node for node, centrality in ranked_nodes[:num_filters_to_prune]]
    
    return ranked_nodes


def filter_pruning_using_ranked_weighted_degree(filters, num_filters_to_prune, ascending=False):
    num_filters, channel_size = filters.shape
    cosine_similarities = cosine_similarity(filters)
    print(cosine_similarities)
    # Create a graph
    G = nx.Graph()
    G.add_nodes_from(range(num_filters))
    
    for i in range(num_filters):
        for j in range(i+1, num_filters):
            G.add_edge(i, j, weight=cosine_similarities[i, j])
    
    # Compute weighted degree centrality
    weighted_degree_centrality = {node: sum([G.edges[node, neighbor]['weight'] for neighbor in G.neighbors(node)]) for node in G.nodes()}
    
    # Rank the nodes
    ranked_nodes = sorted(weighted_degree_centrality.items(), key=lambda x: x[1], reverse=not ascending)
    
    # Identify filters to prune
    filters_to_prune = [node for node, centrality in ranked_nodes[:num_filters_to_prune]]
    
    return ranked_nodes#,filters_to_prune





def filter_pruning_using_ranked_weighted_degree_abs(filters, num_filters_to_prune, ascending=False):
    num_filters, channel_size = filters.shape
    cosine_similarities = cosine_similarity(filters)
    print(cosine_similarities)
    # Create a graph
    G = nx.Graph()
    G.add_nodes_from(range(num_filters))
    
    for i in range(num_filters):
        for j in range(i+1, num_filters):
            G.add_edge(i, j, weight=cosine_similarities[i, j])
    
    # Compute weighted degree centrality
    weighted_degree_centrality = {node: sum([abs(G.edges[node, neighbor]['weight']) for neighbor in G.neighbors(node)]) for node in G.nodes()}
    
    # Rank the nodes
    ranked_nodes = sorted(weighted_degree_centrality.items(), key=lambda x: x[1], reverse=not ascending)
    
    # Identify filters to prune
    filters_to_prune = [node for node, centrality in ranked_nodes[:num_filters_to_prune]]
    
    return ranked_nodes#,filters_to_prune


def Kmeans_filters(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters).fit(data)
    closest_points, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, data)
    return closest_points

def show_scores(scores):
    x_val = [x[0] for x in scores]
    y_val = [1-x[1] for x in scores]

    print(x_val)
    plt.yscale('log')
    # plt.plot(x_val,y_val)
    plt.scatter(x_val,y_val,s=2)
    plt.show()
# The CS-WASPAA function to calculate the cosine similarity weight importance index. 
def CS_WASPAA(Z):
    a,b,c,d=np.shape(Z) # "d" is number of filters, "c" is number of channels, "a" and "b" represent length of the filter.
    A= np.reshape(Z,(-1,c,d)) # reshape filters
    N = np.zeros((a*b,d)) 
    for i in range(d):
        data= A[:,:,i]
        N[:,i]=rank1_apporx(data)
    W= np.zeros((d,d))
    # print(N)
    # It expects filling W or Z directly.
    sim = filter_pruning_using_ranked_weighted_degree_update(N.T,d)      #cosine_similarity(N.T)
    # scores = Kmeans_filters(sim,d)
    # nodes to keep
    return sim#scores


#%% Active filter pruning methods....



def HRank(feature_maps):
    avg_R  = [ ]
    example_R =[ ]
    
    for i in range(np.shape(feature_maps)[1]):
        for n in range(np.shape(feature_maps)[0]):  # nth example ith filter rank..
            example_R.append(np.linalg.matrix_rank(feature_maps[n,i,:,:]))
        avg_R.append(np.average(example_R))   
        example_R = [ ]
    return np.argsort(avg_R),np.sort(avg_R)

def Energy_aware(feature_maps):
    avg_R  = [ ]
    example_R =[ ]
    for i in range(np.shape(feature_maps)[1]):
        for n in range(np.shape(feature_maps)[0]):  # nth example ith filter rank..
            # a,b,c = np.linalg.svd(feature_maps[n,:,:,i])    
            nuclear_norm = np.linalg.norm(features_maps[n,i,:,:],ord='nuc')#np.sum(b)
            example_R.append(nuclear_norm)
        avg_R.append(np.average(example_R))   
        example_R = [ ]
    return np.argsort(avg_R),np.sort(avg_R)


#%%  Load unpruned CNN weights

W_init = list(np.load('/~/DCASE21_Net/unpruned_model_DCASE21_Net_48.58/unpruned_model_weights.npy',allow_pickle=True))


#%% passive filter pruning: Obtain ranked indesex and importance scores for each layer filters and save the ranked indexex per layer

os.chdir('~/DCASE21/important_index/graph_sim_weighted_degree_update/')

#DCASE21
indexes = [0,6,12] # indexes of the desired weight layers
L = [1,2,3] # layer numbers


for qw in range(1):
    for j in range(len(L)):
        W_2D=W_init[indexes[j]]
        W=np.reshape(W_2D,(49,np.shape(W_2D)[2],np.shape(W_2D)[3]))
        print(np.shape(W),'layer  :','  ',L[j])
        imp_list= CS_WASPAA(W_2D)
        print('imp_list',imp_list)
        file_name='sim_index'+str(L[j])+'.npy'
        np.save(file_name,imp_list)
        
#%% active filter pruning indexes

os.chdir('~/important_indexes/PANNS_CNN14/important_index/Energy_aware/')
data = np.load('~/data_for_active/numpy_active_pruning/data_conv4.npy') # number of examples, number of feature maps, length of feature maps (load data)

features_maps =  np.reshape(data,[data.shape[0],data.shape[1],62, 4])
for l in range(1):
    # sorted_index, sorted_rank = HRank(features_maps)
    sorted_index, sorted_rank = Energy_aware(features_maps)
    print(' total fitlers', len(sorted_index))
    file_name ='block4' + '.npy'
    np.save(file_name,sorted_index)



        
