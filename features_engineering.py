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
    tx1 = []
    tx2 = []
    tx3 = []
    
    #l = nb lignes in x and c = nb columns in x
    (l, c) = np.shape(x)
    
    for i in l:
        if(x(i, 22) == 1):
            tx1 = np.vstack([tx1, x(i, :)])
        elseif(x(i, 22) == 2):
            tx2 = np.vstack([tx2, x(i, :)])
        elseif(x(i, 22) == 3):
            tx3 = np.vstack([tx3, x(i, :)])
            
    return tx1, tx2, tx3
    

