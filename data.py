# Traitement des données afin de pouvoir appliquer un modèle dessus
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import csv

with open('Chromatos/CHROMTAB.CSV') as csvfile :
    reader = csv.reader(csvfile)
    k = 0
    csv_files = []

    for i, row in enumerate(reader) :
        if row[0] == 'Path':
            if i != 0 :
                csv_files.append(csv_file)
            csv_file = []
            k = i

        elif i == k+1 :
            name = row[1].strip('.D')
            csv_file.append(name)
            print(name)

        else :
            print(row)
            if i != k+2 :
                print(row[0], csv_file)
                csv_file.append({'Retention time' : row[0], 'Abundance' : row[1]})

for csv_file in csv_files :
    with open(f'Chromatos/{csv_file[0]}.csv', 'w') as csvfile :
        writer = csv.DictWriter(csvfile, fieldnames=csv_file[1].keys())
        writer.writeheader()
        for point in csv_file[1:]:
              writer.writerow(point)
