# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:30:50 2023

@author: macgr
"""

from recup_data_pyms import data
from trait_data_pyms import traitement_tot

def simulation() :
    df_complet = data()

    df = traitement_tot(df_complet)


    return df

if __name__ == '__main__' :
    df = simulation()
