# Traitement des données afin de pouvoir appliquer un modèle dessus
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import csv
import os
import math

def separation_fichier(files) :
    liste_fichier = []
    for file in files :
        file = f'Chromatos/CHROMTAB_{file}.CSV'
        with open(file, encoding = 'ISO-8859-1') as csvfile :
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
                liste_fichier.append(csv_file[0])
                writer = csv.DictWriter(csvfile, fieldnames=csv_file[1].keys())
                writer.writeheader()
                for point in csv_file[1:]:
                    writer.writerow(point)

    return liste_fichier

def creation_df(liste_fichier) :
    liste_df = []

    for fichier in liste_fichier :
        liste_df.append(pd.read_csv(f'Chromatos/{fichier}.csv'))

    return liste_df

def prob_rt(liste_df) :
    taille = [df.shape[0] for df in liste_df]
    ref = np.max(taille)
    df_ref = liste_df[np.argmax(taille)]

    for i, df in enumerate(liste_df) :
        taille = df.shape[0]

        if ref-taille != 0 :
            for j in range(ref-taille) :
                df = pd.concat([df, df.loc[[df.shape[0]-1]]], ignore_index=True)
                liste_df[i] = df

    for df in liste_df :
        df['Retention time'] = df_ref['Retention time']
    return liste_df

def test(liste_df) :
    taille = False
    tps_diff = False
    tps_dup = False

    df_ref = liste_df[0]

    for df in liste_df[1:] :
        if df_ref.shape[0] != df.shape[0] :
            taille = True
    if taille :
        return 'Pb de taille'

    for i, tps in enumerate(df['Retention time']) :
        if df_ref['Retention time'][i] != tps :
            tps_diff = True
        tps_l = []
        if tps in tps_l :
            tps_dup = true

    if tps_diff :
        return 'Pb de tps'

    if tps_diff :
        return 'Pb de tps_l'

    return 'Ok'

def data_base(files) :

    liste_fichier = separation_fichier(files)
    liste_df = creation_df(liste_fichier)

    liste_df = prob_rt(liste_df)

    if test(liste_df) == 'Ok' :
        liste_df[0].set_index('Retention time', inplace = True)
        df_complet = liste_df[0].T

        for df in liste_df[1:] :
            df.set_index('Retention time', inplace = True)
            df_complet = pd.concat([df_complet, df.T], ignore_index=True)

        df_complet.insert(0, 'Name', liste_fichier)
        df_complet.set_index('Name', inplace = True)

        return df_complet

    else :
        return test(liste_df)
