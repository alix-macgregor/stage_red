# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:30:50 2023

@author: macgr
"""

from recup_data_pyms import data
from recup_data_tot import data_mode
from trait_data_pyms import traitement_tot

import pandas as pd

def simulation(tot = False) :
    if tot :

        df_hydro = data_mode('hydro')
        df_alcool = data_mode('alcool')

        print(df_complet.columns)

        df_complet = pd.merge(df_hydro, df_alcool, right_index= True, left_index= True)
        df = traitement_tot(df_complet)

    else :
        df_complet = data()

        df = traitement_tot(df_complet)


    return df

if __name__ == '__main__' :
    df = simulation()
