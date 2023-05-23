# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:39:23 2023

@author: macgr
"""

# Traitement des données afin de pouvoir appliquer un modèle dessus

""" Import des bibliothèques utiles """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import csv

""" Chromatogramme de l'échantillon """

file = "Chromatos/tic_front_ech_1_hydro.csv"
ech_1_hydro_df = pd.read_csv(file, skiprows=1)

plt.plot(ech_1_hydro_df['Start of data points'], ech_1_hydro_df['Area'])

# Note : J'ai récupéré tous les chromatogrammes normalement! Essayer de voir comment les séparer.

""" Ensemble des chromatogrammes """
# J'ai dû faire en deux fois à voir si ça marche ou pas...

with open('Chromatos/CHROMTAB.CSV') as csvfile :
    reader = csv.reader(csvfile)
    k = 0
    csv_files = []
    csv_file = []
    
    for i, row in enumerate(reader) :
        if row[0] == 'Path':
            if i != 0 :
                csv_files.append(csv_file)
            csv_file = []
            k = i
            
        elif i == k+1 :
            name = row[1].strip('.D')
            csv_file.append(name)
        
        else :
            if i != k+2 :
                csv_file.append({'Retention time' : row[0], 'Abundance' : row[1]})
                

for csv_file in csv_files :
    with open(f'Chromatos/{csv_file[0]}.csv', 'w') as csvfile :
        writer = csv.DictWriter(csvfile, fieldnames=csv_file[1].keys())
        writer.writeheader()
        for point in csv_file[1:]:
              writer.writerow(point)