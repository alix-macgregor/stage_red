# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:30:50 2023

@author: macgr
"""

from recup_data import data_mode, data_test
from trait_data import traitement_tot

import pandas as pd

def simulation() :
    """
    Lance la simulation
    """

    # récupération de la base de données connue
    df_hydro = data_mode('hydro')
    df_alcool = data_mode('alcool')

    # récupération de le base de donnée inconnue
    df_hydro_test = data_test('hydro')
    df_alcool_test = data_test('alcool')

    # traitement des échantillons inconnus
    liste_df = traitement_tot(df_hydro, df_alcool, df_hydro_test, df_alcool_test)

    liste_df_tot = traitement_tot(df_hydro, df_alcool, tot = True)
    return liste_df, liste_df_tot

if __name__ == '__main__' :
    liste_df, liste_df_tot = simulation()
