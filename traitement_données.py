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
import seaborn as sns

import os
import csv
import math

# PyMS
from pyms.GCMS.IO.ANDI import ANDI_reader
from pyms.IntensityMatrix import build_intensity_matrix
from pyms.BillerBiemann import BillerBiemann
from pyms.Peak.Function import peak_sum_area

""" Chromatogramme de l'échantillon """

def recup_ech(dir) :
    datas = []
    for name in os.listdir(dir) :
        andi_file = dir + "/" + name
        raw_data = ANDI_reader(andi_file)

        datas.append(raw_data)

    return datas

def recup_areas(datas) :
    data_peak = []
    for data in datas :
        im = build_intensity_matrix(data)
        peak_list = BillerBiemann(im)
        data_peak.append(peak_list)
    areas = []
    for peak in peak_list :
        area = peak_sum_area(im, peak)
        areas.append((peak.rt, area))

    return areas

def traitement_ANDI(dir) :
    datas = recup_ech(dir)
    areas = recup_areas(datas)
