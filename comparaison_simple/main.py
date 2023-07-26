# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:57:04 2023

@author: macgr
"""

from recup_data import data_base
from trait_data import traitement

import matplotlib.pyplot as plt

def simulation(ech = None) :
    df_complet, types = data_base(ech)
    
    if types is None :
        print("Erreur : le test taille/rt n'est pas passé")
        return df_complet, None, None
    
    df_n, df_pca, df_complet = traitement(df_complet, ech)
    
    plt.plot(df_complet.columns, df_complet.loc[ech])
    plt.plot(df_complet.columns, -df_complet.loc[df_pca.index[1]])
    
    plt.title(f'{ech} VS {df_pca.index[1]}')
    plt.xlabel('Time')
    plt.ylabel('Abundance')
    plt.show()
    
    
    print(df_pca.head())  

    return df_n, df_pca, df_complet

if __name__ == '__main__' :
    ech = 'ech270_c'
    df_n, df_pca, df_complet = simulation(ech)
    
    ech = 'ech91_b'
    df_n, df_pca, df_complet = simulation(ech)
    
    ech = 'ech122_120'
    df_n, df_pca, df_complet = simulation(ech)
   
    