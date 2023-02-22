#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv #lecture du fichier data
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def extract(fichier):
    X = []
    y = []
    with open(fichier, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            y.append(row[17]) #status : 0 ou 1
            x1 = row[1:17]
            x2 = row[18:]
            X.append(x1 + x2) #tous les autres attributs
    return X[1:],y[1:]

X,y = extract('./data/Parkinsons.data')
