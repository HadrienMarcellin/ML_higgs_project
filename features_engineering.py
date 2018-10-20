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


