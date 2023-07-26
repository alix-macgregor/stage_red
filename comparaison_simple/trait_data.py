# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:00:15 2023

@author: macgr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity

def normal(df_complet) :
    df_normal = df_complet.drop(columns = 'Type')

    for feature in df_normal.columns :
        minmax_scaler= MinMaxScaler().fit(df_complet[[feature]])
        
        df_normal[feature] = minmax_scaler.transform(df_normal[[feature]])
        
    print('ok normal')

    return df_normal

def proj(df_normal) :
    pca = PCA().fit(df_normal)

    df_proj = pca.transform(df_normal)

    df_proj = pd.DataFrame(df_proj, columns = [f'PC{i}' for i in range(1, df_proj.shape[1]+1)])

    df_proj.insert(0, 'Name', df_normal.index)
    df_proj.set_index('Name', inplace = True)
    
    print('ok proj')

    return df_proj

def run_kmeans(X, n_clusters = int) :
    kmeans = MiniBatchKMeans(n_clusters = n_clusters, random_state = 0)
    kmeans.fit(X)
    labels_kmeans = kmeans.predict(X)
    X['cluster_kmeans'] = labels_kmeans
    
    print('ok kmeans')
    
    return X

def apply_cosine (data, ech) :
    v1 = np.array(data.loc[ech]).reshape(1, -1)
    sim1 = cosine_similarity(data, v1).reshape(-1)
    dictDf = {'Chromatos Sims': sim1 }
    recommendation_df = pd.DataFrame(dictDf, index = data.index)
    
    print('ok cosine')
    
    return recommendation_df

def merge_cosinedf (df, data, ech) :
    df_copy = df.drop(index = ech)
    results = pd.merge(data, df_copy, left_index = True, right_index = True).sort_values(by = "Chromatos Sims", ascending = False)
    df_ech = pd.merge(data, df.loc[[ech]], left_index = True, right_index = True).sort_values(by = "Chromatos Sims", ascending = False)
    results = pd.concat([df_ech, results])
    
    print('ok merge')
    
    return results[['Chromatos Sims', 'Type']]

def traitement(df_complet) :
    df_normal = normal(df_complet)
    df_proj = proj(df_normal)

    data_n = run_kmeans(df_normal, 3)
    data_pca = run_kmeans(df_proj, 3)

    for ech in df_complet.index :
        df_n = apply_cosine(data_n, ech)
        df_pca = apply_cosine(data_pca, ech)
    
        df_n = merge_cosinedf(df_n, df_complet, ech)
        df_pca = merge_cosinedf(df_pca, df_complet, ech)

        print(df_pca.head())
        
        df_complet.rename(columns = lambda x : float(x))

    return df_n, df_pca, df_complet
