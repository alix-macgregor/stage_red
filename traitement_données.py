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