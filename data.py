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
        file = liste_df[i]
        df = pd.read_csv(file)
        taille = df.shape[0]

        if taille_ref-taille != 0 :
            prob.append(True)

        else :
            prob.append(False)

    return prob

def rajout_ligne(liste_df) :
    prob = prob_taille(liste_df)

    for i, df in enumerate(liste_df) :
        if prob[i] :
            df = pd.concat([df, df.loc[[df.shape[0]-1]]], ignore_index=True)

    return liste_df
