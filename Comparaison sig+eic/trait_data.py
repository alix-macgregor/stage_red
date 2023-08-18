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

    # print('ok normal')

    return df_normal

def encode(df_complet) :
    """
    Cette fonction permet de transformer les catégories (A, DP, S, etc...) en chiffres.
    """
    target = df_complet['Type']

    label_encoder = LabelEncoder().fit(target)

    df_encode = df_complet.copy()
    df_encode['Type_transform'] = label_encoder.transform(target)

    # print('ok encode')

    return df_encode, label_encoder

def lda(df_complet) :
    """
    Cette fonction permet d'appliquer l'analyse du discrimant linéaire
    """

    df_complet, label_encoder = encode(df_complet)

    df_complet.columns = [str(x) if x != 'Type' else x for x in df_complet.columns ]

    # Liste des échantillons dont on connaît déjà le type
    df_lda = df_complet.drop(columns = 'Type')[df_complet['Type'] != 'Unknown']

    # Liste des échantillons inconnus pour lesquels il faut prédire le type avant
    X_pred = df_complet.drop(columns = ['Type', 'Type_transform'])[df_complet['Type'] == 'Unknown']

    X = df_lda.drop(columns = 'Type_transform')
    y = df_lda['Type_transform']

    # On applique la LDA à notre problème de classification
    lda = LinearDiscriminantAnalysis().fit(X, y)

    columns = lda.get_feature_names_out()

    df_complet_lda = pd.DataFrame(lda.transform(X), index = X.index, columns = columns)

    type_pred = lda.predict(X_pred)

    pred = label_encoder.inverse_transform(type_pred)

    X_pred = pd.DataFrame(lda.transform(X_pred), index = X_pred.index, columns = columns)

    df_complet_lda = pd.concat([df_complet_lda, X_pred])

    df_complet_lda.sort_index(key=lambda x: np.argsort(index_natsorted(df_complet_lda.index)), inplace = True)

    df_complet_lda['Type'] = df_complet['Type']

    # print('ok lda')

    return df_complet_lda, pred[0]

def apply_cosine (df_complet, ech) :
    """
    Cette fonction permet de comparer un échantillon à tous les autres et de déterminer les coefficients de similitude
    """
    X = df_complet.drop(columns = ['Type'])


    v1 = np.array(X.loc[ech]).reshape(1, -1)

    sim1 = cosine_similarity(X, v1).reshape(-1)
    dictDf = {'Chromatos Sims': sim1 }
    recommendation_df = pd.DataFrame(dictDf, index = df_complet.index)

    # print('ok cosine')

    return recommendation_df

def merge_cosinedf (recommendation_df, df_complet, ech) :
    """
    Cette fonction permet de regrouper les recommendations
    """
    df_copy = recommendation_df.drop(index = ech)
    results = pd.merge(df_complet, df_copy, left_index = True, right_index = True).sort_values(by = "Chromatos Sims", ascending = False)
    df_ech = pd.merge(df_complet, recommendation_df.loc[[ech]], left_index = True, right_index = True).sort_values(by = "Chromatos Sims", ascending = False)
    results = pd.concat([df_ech, results])

    # print('ok merge')

    return results[['Chromatos Sims', 'Type']]

def traitement(df_hydro, df_alcool, df_hydro_test, df_alcool_test, tot) :
    """
    Cette fontion permet de regrouper tout le traitement des données et renvoie le dataframe avec les similitudes.
    """

    if tot == True :
        df_complet = pd.merge(df_hydro, df_alcool, right_index= True, left_index= True)

        df_complet['Type'] = df_complet['Type_y']
        df_complet.drop(columns=['Type_x', 'Type_y'], inplace=True)

        liste_df = []

        for i, ech in enumerate(df_complet.index) :
            print(f'\n{ech} :\n')

            classe = df_complet.loc[[ech]]['Type']
            df_complet.loc[ech, 'Type'] = 'Unknown'

            df_normal = normal(df_complet)

            df_lda, pred = lda(df_normal)

            print(f"\nPour l'échantillon {ech}, la prédiction lda est {pred}\n")

            df_reco = apply_cosine(df_lda, ech)

            df = merge_cosinedf(df_reco, df_hydro, ech)

            print(df.head())

            df_complet.loc[ech, 'Type'] = classe[0]

            liste_df.append(df)

        return liste_df

    else :

        liste_df = []

        for ech in df_hydro_test.index :
            print(f'\n{ech}\n')

            df_hydro_copy = pd.concat([df_hydro, df_hydro_test.loc[[ech]]])
            df_alcool_copy = pd.concat([df_alcool, df_alcool_test.loc[[ech]]])

            df_complet = pd.merge(df_hydro_copy, df_alcool_copy, right_index= True, left_index= True)

            df_complet['Type'] = df_complet['Type_y']
            df_complet.drop(columns=['Type_x', 'Type_y'], inplace=True)

            df_normal = normal(df_complet)

            df_lda, pred = lda(df_normal)

            print(f"\nPour l'échantillon {ech}, la prédiction lda est {pred}\n")

            df_reco = apply_cosine(df_lda, ech)

            df = merge_cosinedf(df_reco, df_hydro, ech)

            print(df.head())

            liste_df.append(df)

        return liste_df

def traitement_tot(df_hydro, df_alcool, df_hydro_test = None, df_alcool_test = None, tot = False) :
    """
    Cette fonction permet de faire le traitement pour tous les échantillons inconnus
    """
    if tot :
        message = traitement(df_hydro, df_alcool, df_hydro_test, df_alcool_test, tot)
        return message

    else :
        liste_df = traitement(df_hydro, df_alcool, df_hydro_test, df_alcool_test, tot)

        if liste_df == [] :
            print("Vous n'avez pas téléchargé de nouveaux échantillons à comparer")

        return liste_df
