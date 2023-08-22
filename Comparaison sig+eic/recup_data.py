# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:08:28 2023

@author: macgr
"""

import numpy as np
import pandas as pd

import os

from natsort import index_natsorted

from pyms.GCMS.IO.ANDI import ANDI_reader
from pyms.IntensityMatrix import build_intensity_matrix
from pyms.eic import build_extracted_intensity_matrix

import iisignature as isig


def recuperation_ech(file) :
    """
    Cette fonction permet de récupérer les échantillons enregistrés dans le
    dossier Echantillons_AIA et d'en extraire les informations grâce au module
    PyMS.
    """

    chrom = ANDI_reader(file)
    return chrom

def eic(chrom, liste_ion) :
    """
    Cette fonction permet d'extraire les chromatogrammes relatifs de chaque ions
    sélectionnés et d'en créer une liste.
    """

    # Création de la matrice d'intensité de tout le chromatogramme
    im = build_intensity_matrix(chrom)

    # extraction des eics
    eic = build_extracted_intensity_matrix(im,
                                           liste_ion,
                                           right_bound = 0.2,
                                           left_bound = 0.2)

    # Récupération des eics sous forme de liste
    eic_im = eic.intensity_matrix

    # Normalisation des eics par rapport à la somme de toutes les intensités
    eic_im_list = []
    for array in eic_im.T :
        item_list = []
        somme = float(np.sum(np.array(array)))
        for item in array :
            if somme == 0 :
                item_list.append(item)
            else :
                norm = float(item/somme)
                item_list.append(norm)

        eic_im_list.append(item_list)
    return eic_im_list

def signature(X, k) :
    """
    Cette fonction permet de calculer la signature d'ordre k de la liste de eic.
    """

    # si k = 0 la signature vaut 1
    if k == 0 :
        return np.full((np.shape(X[0])[0], 1), 1)

    # sinon il faut rajouter le 1 au début et utiliser la fonction sig de iisig
    else :
        sigX = []
        for v in X :
            v = np.array(v)
            v = v.T
            d = v.shape[1]
            sig = np.zeros((isig.siglength(d,k) + 1))

            sig[0] = 1
            sig[1:] = isig.sig(v, k)
            sigX.append(sig)

        return sigX

def dataframe(sig, ech, typ) :
    """
    Cette fonction permet de créer un dataframe à partir de la signature
    calculée précédemment.
    """
    df = pd.DataFrame(sig, columns = [ech])
    df = df.T

    df['Type'] = [typ]

    return df

def enregistrement(df_complet, mode) :
    """
    Cette fonction permet d'enregistrer les signatures telles quelles dans un
    csv disponible dans le fichier Stage - pas besoin de refaire le calcul des
    signatures de manière systématique.
    """
    mode = '_' + mode

    df_complet.to_csv(f'Echantillons{mode}.csv', index = True, header = True)

def data_test(mode) :
    """
    Permet de créer le dataframe avec les données à tester (permet de faire le retraitement)
    """

    if mode == 'alcool' :
        liste_ion = [43, 31, 29, 73, 87, 55, 74]

    elif mode == 'hydro' :
        liste_ion = [43, 83, 57, 71, 128, 142, 104, 118, 91, 105, 134, 117, 149]

    liste_ech = []
    liste_type = []
    X = []
    df_complet = None

    # Si le fichier n'existe pas : on traite tous les échantillons
    for fil in os.listdir('Echantillons_AIA/Chromatos_Tester.AIA') :
        if f'{mode}.CDF' in fil and 'Zone' not in fil :

            file = f'Echantillons_AIA/Chromatos_Tester.AIA/{fil}'
            fil = fil.rstrip(f'{mode}.CDF')
            fil = fil.strip('_')
            fil = fil.lower()
            fil = fil.replace(' ', '')
            fil = fil.replace('-', '_')

            try :
                chrom = recuperation_ech(file)
                if chrom.min_rt < 149 and mode == 'hydro' :
                    print('trim')
                    chrom.trim("149s", "2969s")
                eic_im_list = eic(chrom, liste_ion)

            except ValueError :
                print(f'pb recup {fil}')
                eic_im_list = [[0 for i in range(12718)] for j in range(len(liste_ion))]


            X.append((eic_im_list))
            liste_ech.append(fil)
            liste_type.append('Unknown')

    sigs = signature(X, 3)

    # On crée le dataframe complet
    for i, sig in enumerate(sigs) :
        if df_complet is None :
            df_complet = dataframe(sig, liste_ech[i], liste_type[i])

        else :
            df_complet = pd.concat([df_complet,
                                    dataframe(sig,
                                                liste_ech[i],
                                                liste_type[i])])
    return df_complet

def data_mode(mode) :
    """
    Cette fonction permet de regrouper toutes les fonctions pour traiter tous
    les échantillons.
    """

    if mode == 'alcool' :
        liste_ion = [43, 31, 29, 73, 87, 55, 74]

    elif mode == 'hydro' :
        liste_ion = [43, 83, 57, 71, 128, 142, 104, 118, 91, 105, 134, 117, 149]

    liste_classe = ['DP', 'S', 'O', 'NP', 'NA', 'D', 'I', 'In', 'PI']
    liste_ech = []
    liste_type = []
    X = []
    df_complet = None

    mode_file = '_' + mode

    # Si le fichier n'existe pas : on traite tous les échantillons
    if f'Echantillons{mode_file}.csv' not in os.listdir() :
        for classe in liste_classe :
            for fil in os.listdir(f'Echantillons_AIA/Chromatos_{classe}.AIA') :

                if f'{mode}.CDF' in fil and 'Zone' not in fil :

                    file = f'Echantillons_AIA/Chromatos_{classe}.AIA/{fil}'
                    fil = fil.rstrip(f'{mode}.CDF')
                    fil = fil.strip('_')
                    fil = fil.lower()
                    fil = fil.replace(' ', '')

                    try :
                        chrom = recuperation_ech(file)
                        eic_im_list = eic(chrom, liste_ion)

                    except ValueError :
                        print(f'pb recup {fil}')


                    X.append((eic_im_list))
                    liste_ech.append(fil)
                    liste_type.append(classe)

        sigs = signature(X, 3)

        # On crée le dataframe complet
        for i, sig in enumerate(sigs) :

            if df_complet is None :

                df_complet = dataframe(sig, liste_ech[i], liste_type[i])

            else :

                df_complet = pd.concat([df_complet,
                                        dataframe(sig,
                                                  liste_ech[i],
                                                  liste_type[i])])

    # Si le fichier existe, on récupère le dataframe enregistré sous csv
    else :
        df_complet = pd.read_csv(f'Echantillons{mode_file}.csv', index_col = 0)
        df_complet.columns = [int(x) if x != 'Type' else x for x in df_complet.columns ]

        for classe in liste_classe :
            for fil in os.listdir(f'Echantillons_AIA/Chromatos_{classe}.AIA') :

                if f'{mode}.CDF' in fil and 'Zone' not in fil:

                    file = f'Echantillons_AIA/Chromatos_{classe}.AIA/{fil}'
                    fil = fil.rstrip(f'{mode}.CDF')
                    fil = fil.strip('_')
                    fil = fil.lower()
                    fil = fil.replace(' ', '')

                    if fil not in df_complet.index :

                        try :
                            chrom = recuperation_ech(file)
                            eic_im_list = eic(chrom, liste_ion)

                        except ValueError :
                            print(f'pb recup {fil}')
                            print(np.array(eic_im_list).shape)

                        liste_ech.append(fil)
                        liste_type.append(classe)

                        X.append(eic_im_list)

        sigs = signature(X, 3)

        # On rajoute les nouvelles signatures au dataframe
        for i, sig in enumerate(sigs) :
            df_complet = pd.concat([df_complet,
                                    dataframe(sig, liste_ech[i], liste_type[i])])

    df_complet.sort_index(key=lambda x: np.argsort(index_natsorted(df_complet.index)), inplace = True)

    enregistrement(df_complet, mode)

    return df_complet
