# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 11:04:05 2025

@author: michel
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_total = pd.read_csv("dwt_feat_data.csv",header=None) #stats_features_mfcc_dwt_rdft_feat _data.csv

X = data_total.iloc[:, :-1].values   
y = data_total.iloc[:, -1].values   

data_total1 = pd.read_csv("rdft_feat_data.csv",header=None) #stats_features_mfcc_dwt_rdft_feat _data.csv

X1 = data_total1.iloc[:, :-1].values   
y1 = data_total1.iloc[:, -1].values 

# Trouver les classes uniques
classes = np.unique(y)

for c in classes[:3]:
    # Trouver un index correspondant à cette classe
    idx = np.where(y == c)[0][0]
    sample = X[idx]
    
    # Affichage
    plt.figure(figsize=(8, 4))
    plt.plot(sample)
    plt.title(f"Classe {c} - Exemple de sample")
    plt.xlabel("Index de feature")
    plt.ylabel("Valeur")
    plt.grid(True)
    plt.tight_layout()
    plt.show()