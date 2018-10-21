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
        #c = np.c_[tx, sinfeature]
        #tx = c.copy()
        tx[:, i] = sinfeature
    
    return tx

#####################################  --  ARCSIN -- ###################################

def transform_feature_arcsin(x, features):
    tx = x.copy()
    for i in features:
        feature = tx[:,i]
        arcsinfeature = np.arcsin(feature)
        tx[:, i] = arcsinfeature
    
    return tx

#####################################  --  COS -- ###################################

def transform_feature_cos(x, features):
    tx = x.copy()
    for i in features:
        feature = tx[:,i]
        cosfeature = np.cos(feature)
        #add the new feature at the end !
        #c = np.c_[tx, cosfeature]
        #tx = c.copy()
        tx[:, i] = cosfeature
    
    return tx

#####################################  --  POWER2 -- ###################################

def transform_feature_power(x, features, power):
    tx = x.copy()
    for i in features:
        feature = tx[:,i]
        featurepower = np.power(feature, power)
        tx[:,i] = featurepower
        
    return tx

#####################################  --  SEPARATE PRI_jet_num -- ###################################

def separate_PRI_jet_num(x, y):
    (l, c) = np.shape(x)
    
    tx0 = np.empty((0,c), int)
    tx1 = np.empty((0,c), int)
    tx2 = np.empty((0,c), int)
    tx3 = np.empty((0,c), int)
    y0 = np.empty((0,), int)
    y1 = np.empty((0,), int)
    y2 = np.empty((0,), int)
    y3 = np.empty((0,), int)

    for i in range(l):
        if(x[i, 22] == 0.0):
            tx0 = np.append(tx0, [x[i,:]], axis=0)
            y0 = np.append(y0, [y[i]], axis=0)
        elif(x[i, 22] == 1.0):
            tx1 = np.append(tx1, [x[i,:]], axis=0)
            y1 = np.append(y1, [y[i]], axis=0)
        elif(x[i, 22] == 2.0):
            tx2 = np.append(tx2, [x[i,:]], axis=0)
            y2 = np.append(y2, [y[i]], axis=0)
        elif(x[i, 22] == 3.0):
            tx3 = np.append(tx3, [x[i,:]], axis=0)
            y3 = np.append(y3, [y[i]], axis=0)
            
    return tx0, y0, tx1, y1, tx2, y2, tx3, y3
    

