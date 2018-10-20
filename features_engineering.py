# -*- coding: utf-8 -*-
"""machine learning functions for project 1."""
import numpy as np


##################################### -- LOG -- ######################################

def transform_feature_log(x, features):
    tx = x.copy()
    for i in features:
        feature = tx[:,i] + abs(np.min(tx[:,i])) + 0.1 #pour enlever les valeurs négatives
        logfeature = np.log(feature)
        tx[:, i] = logfeature
    
    return tx


#####################################  --  SIN -- ###################################

def transform_feature_sin(x, features):
    tx = x.copy()
    for i in features:
        feature = tx[:,i]
        sinfeature = np.sin(feature)
        #add the new feature at the end !
        c = np.c_[tx, sinfeature]
        tx = c.copy()
    
    return tx

#####################################  --  COS -- ###################################

def transform_feature_cos(x, features):
    tx = x.copy()
    for i in features:
        feature = tx[:,i]
        cosfeature = np.cos(feature)
        #add the new feature at the end !
        c = np.c_[tx, cosfeature]
        tx = c.copy()
    
    return tx

#####################################  --  SEPARATE PRI_jet_num -- ###################################

def separate_PRI_jet_num(x):
    (l, c) = np.shape(x)
    
    tx0 = np.empty((0,c), int)
    tx1 = np.empty((0,c), int)
    tx2 = np.empty((0,c), int)
    tx3 = np.empty((0,c), int)

    for i in range(l):
        if(x[i, 22] == 0.0):
            tx0 = np.append(tx0, [x[i,:]], axis=0)
        elif(x[i, 22] == 1.0):
            tx1 = np.append(tx1, [x[i,:]], axis=0)
        elif(x[i, 22] == 2.0):
            tx2 = np.append(tx2, [x[i,:]], axis=0)
        elif(x[i, 22] == 3.0):
            tx3 = np.append(tx3, [x[i,:]], axis=0)
            
    return tx0, tx1, tx2, tx3
    

