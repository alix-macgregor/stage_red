# Traitement des données afin de pouvoir appliquer un modèle dessus
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import csv

def separation_fichier(file) :
    liste_fichier = []
    with open(file) as csvfile :
        reader = csv.reader(csvfile)
        k = 0
        csv_files = []

        for i, row in enumerate(reader) :
            if row[0] == '\ufeffPath' or row[0] == 'Path' :
                if i != 0 :
                    csv_files.append(csv_file)
                csv_file = []
                k = i

            elif i == k+1 :
                name = row[1].strip('.D')
                if '-' in name :
                    name = name.replace("-", "_")
                csv_file.append(name)

            else :
                if i >= k+3 :
                    csv_file.append({'Retention time' : row[0], 'Abundance' : row[1]})

    for csv_file in csv_files :
        with open(f'Chromatos/{csv_file[0]}.csv', 'w') as csvfile :
            liste_fichier.append(f'Chromatos/{csv_file[0]}.csv')
            writer = csv.DictWriter(csvfile, fieldnames=csv_file[1].keys())
            writer.writeheader()
            for point in csv_file[1:]:
                  writer.writerow(point)

    return liste_fichier

def creation_df(liste_fichier) :
    liste_df = []

    for fichier in liste_fichier :
        liste_df.append(pd.read_csv(fichier))

    return liste_df

def prob_taille(liste_df) :
    prob = []
    df_ref = liste_df[0]
    taille_ref = df_ref.shape[0]

    for i in range(len(liste_df)) :
        df = liste_df[i]
        taille = df.shape[0]

        if taille_ref-taille != 0 :
            prob.append((True, taille_ref-taille))

        else :
            prob.append((False, taille_ref-taille))

    return prob

def rajout_ligne(liste_df) :
    prob = prob_taille(liste_df)

    for i, df in enumerate(liste_df) :
        pb, delta = prob[i]
        if pb :
            if delta>0 :
                for i in range(delta) :
                    liste_df[i] = pd.concat([df, df.loc[[df.shape[0]-1]]], ignore_index=True)
    return liste_df

def retrait_ligne(liste_df) :
    pass

def tps_retention(liste_df) :
    diff = []
    df_ref = liste_df[0]

    for df in liste_df :
        for i, tps in enumerate(df['Retention time']) :
            if df_ref['Retention time'][i] != tps :
                diff.append(abs(df_ref['Retention time'][i]-tps))

        if np.mean(diff) < 0.004 :
            for i, df in enumerate(liste_df) :
                df['Retention time'] = df_ref['Retention time']
                liste_df[i] = df

        else :
            print("Ecart trop important")
        diff = []

    return liste_df
