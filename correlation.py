# -*- coding: utf-8 -*-
"""machine learning functions for project 1."""
import numpy as np

def pearson_correlation(x, y, num_feature):
    tx = x.copy()
    ty = y.copy()
    xf = tx[:, num_feature]
    xf = standardize(xf)
    x_mean = mean(xf)
    y_mean = mean(ty)
    somme_nom = 0
    somme_denom = 0
    
    
    for i in range(len(y)):
        somme_nom = somme_nom + ((xf[i]-x_mean)*(ty[i]-y_mean))
        somme_denom = somme_denom + (((xf[i]-x_mean)**2)*((ty[i]-y_mean)**2))
        
    pearson_correl = somme_nom/((somme_denom)**(1/2))
    
    return pearson_correl

def corr_pear(y, tx, col):
    mean_y = np.mean(y)
    mean_x = np.mean(tx[col,:])
    numerateur = np.dot((tx[:, col] - mean_x),(y - mean_y))
    denominateur = np.sum((tx[:, col] - mean_x)**2) * np.sum((y - mean_y)**2)
    return numerateur / np.sqrt(denominateur)


def mean(vector):
    somme = 0
    for i in range(len(vector)):
        somme = somme + vector[i]
        
    mean = somme/len(vector)
    return mean

    
def standardize(x):
    """Standardize the original data set."""
    mean_x = mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x 