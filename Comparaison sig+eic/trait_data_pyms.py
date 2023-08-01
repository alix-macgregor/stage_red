# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:00:15 2023

@author: macgr
"""

import numpy as np
import pandas as pd

from natsort import index_natsorted

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def normal(df_complet) :
    """
    Cette fonction permet de normaliser les données
    """

    df_normal = df_complet.drop(columns = 'Type')

    # Min Max Scaler ici, car les données ne sont pas normales
    for feature in df_normal.columns :
        minmax_scaler = MinMaxScaler().fit(df_normal[[feature]])

        df_normal[feature] = minmax_scaler.transform(df_normal[[feature]])

    df_normal['Type'] = df_complet['Type']

    print('ok normal')

    return df_normal

def encode(df_complet) :
    """
    Cette fonction permet de transformer les catégories (A, DP, S, etc...) en chiffres.
    """
    target = df_complet['Type']

    label_encoder = LabelEncoder().fit(target)

    df_encode = df_complet.copy()
    df_encode['Type_transform'] = label_encoder.transform(target)

    print('ok encode')

    return df_encode

def lda(df_complet) :
    """
    Cette fonction permet d'appliquer l'analyse du discrimant linéaire
    """

    df_complet = encode(df_complet)

    # Liste des échantillons dont on connaît déjà le type
    df_lda = df_complet.drop(columns = 'Type')[df_complet['Type'] != 'Unknown']

    # Liste des échantillons inconnus pour lesquels il faut prédire le type avant
    X_pred = df_complet.drop(columns = ['Type', 'Type_transform'])[df_complet['Type'] == 'Unknown']

    X = df_lda.drop(columns = 'Type_transform')
    y = df_lda['Type_transform']

    # On applique la LDA à notre problème de classification
    lda = LinearDiscriminantAnalysis().fit(X, y)



    df_complet_lda = pd.DataFrame(lda.transform(X), index = X.index, columns = ['LDA_1', 'LDA_2'])
    df_complet_lda['Type_transform'] = y

    if df_complet_lda.shape[0] != df_complet.shape[0] :

        type_pred = lda.predict(X_pred)

        X_pred = pd.DataFrame(lda.transform(X_pred), index = X_pred.index, columns = ['LDA_1', 'LDA_2'])
        X_pred['Type_transform'] = type_pred

        df_complet_lda = pd.concat([df_complet_lda, X_pred])

    df_complet_lda.sort_index(key=lambda x: np.argsort(index_natsorted(df_complet_lda.index)), inplace = True)

    df_complet_lda['Type'] = df_complet['Type']

    print('ok lda')

    return df_complet_lda

def apply_cosine (df_complet, ech) :
    """
    Cette fonction permet de comparer un échantillon à tous les autres et de déterminer les coefficients de similitude
    """
    X = df_complet.drop(columns = ['Type'])


    v1 = np.array(X.loc[ech]).reshape(1, -1)

    sim1 = cosine_similarity(X, v1).reshape(-1)
    dictDf = {'Chromatos Sims': sim1 }
    recommendation_df = pd.DataFrame(dictDf, index = df_complet.index)

    print('ok cosine')

    return recommendation_df

def merge_cosinedf (recommendation_df, df_complet, ech) :
    """
    Cette fonction permet de regrouper les recommendations
    """
    df_copy = recommendation_df.drop(index = ech)
    results = pd.merge(df_complet, df_copy, left_index = True, right_index = True).sort_values(by = "Chromatos Sims", ascending = False)
    df_ech = pd.merge(df_complet, recommendation_df.loc[[ech]], left_index = True, right_index = True).sort_values(by = "Chromatos Sims", ascending = False)
    results = pd.concat([df_ech, results])

    print('ok merge')

    return results[['Chromatos Sims', 'Type']]

def traitement(df_complet, ech = None) :
    """
    Cette fontion permet de regrouper tout le traitement des données et renvoie le dataframe avec les similitudes.
    """
    if ech is None :
        return "Vous n'avez pas choisi d'échantillon à comparer"

    # Version choisie du traitement des données :

    df_normal = normal(df_complet)

    df_lda = lda(df_normal)
    df_reco = apply_cosine(df_lda, ech)
    df = merge_cosinedf(df_reco, df_complet, ech)

    print(df.head())

    return df

def traitement_tot(df_complet) :
    """
    Cette fonction permet de faire le traitement v3 pour tous les échantillons
    """

    liste_df_v3 = []

    df_normal = normal(df_complet)

    df_lda = lda(df_normal)

    for ech in df_complet.index :
        if ech != 'ech91' :
            df_reco = apply_cosine(df_lda, ech)
            df_v3 = merge_cosinedf(df_reco, df_complet, ech)

            print(df_v3.head())

            liste_df_v3.append(df_v3)

    return liste_df_v3
