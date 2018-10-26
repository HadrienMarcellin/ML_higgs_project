# -*- coding: utf-8 -*-
"""machine learning functions for project 1."""
import numpy as np



##################################### -- BOSONS AND NAN -- ######################################

def NAN_and_bosons(tx, y, features):

    NAN_boson = np.empty((0,), int)
    NAN_no = np.empty((0,), int)
    NAN_nb = np.empty((0,), int)
    x = tx.copy()
    (l,c)=np.shape(x)
    #boucle for qui circule sur les features

    for j in features:
        nb_bosons = 0
        nb_no = 0
        nb_nan = 0
        #boucle for qui circule sur les samples
        for i in range(l):
            if(x[i,j] <= -999):
                nb_nan = nb_nan + 1
                #regarde si label = boson
                if(y[i] == 1):
                    nb_bosons = nb_bosons + 1
                elif(y[i] == -1):
                    nb_no = nb_no + 1
                
        NAN_boson = np.append(NAN_boson, [nb_bosons], axis=0)
        NAN_no = np.append(NAN_no, [nb_no], axis=0)
        NAN_nb = np.append(NAN_nb, [nb_nan], axis=0) 

    #boucle for qui itère pour connaître le nombre de bosons
    tot_bosons = 0
    for b in range(l):
        if(y[b] == 1):
            tot_bosons = tot_bosons + 1

    for j in range(len(features)):
        print("features {x} has {NumberNan} nan values with {NumberBoson} bosons (out of {tot_bos} total bosons) and {NumberNo} others".format(x=features[j], NumberNan=NAN_nb[j], NumberBoson=NAN_boson[j], tot_bos = tot_bosons, NumberNo=NAN_no[j]))


##################################### -- STANDARDIZE ANGLES -- #################################

def standardize_angles(tx, features):
    
    x = tx.copy()
    
    for i in features:
        standfeature = x[:,i]
        standfeature = (standfeature - np.nanmin(standfeature) - np.pi)/ (np.nanmax(standfeature) - np.nanmin(standfeature)) * 2 * np.pi
        x[:,i] = standfeature
    
    return x

##################################### -- STANDARDIZE Features -- #################################

def standardize_features_according_to_train_set(tx, features, mean, std):
    
    x = tx.copy()

    for i, feature in enumerate(features):
        
        standfeature = x[:,feature]
        standfeature = (standfeature - mean[i])/ std[i]
        x[:,feature] = standfeature
    
    return x


##################################### -- STANDARDIZE Features -- #################################

def standardize_features(tx, features):
    
    mean = []
    std = []
    x = tx.copy()

    
    for i in features:
        
        standfeature = x[:,i]
        std.append(np.nanstd(standfeature))
        mean.append(np.nanmean(standfeature))
        
        standfeature = (standfeature - mean[-1])/ std[-1]
        x[:,i] = standfeature
    
    return x, mean, std 


##################################### -- LOG -- ######################################

def transform_feature_log(x, features):
    tx = x.copy()
    for i in features:
        feature = tx[:,i] - np.nanmin(tx[:,i]) + 0.1 #pour enlever les valeurs négatives
        logfeature = np.log(feature)
        tx[:, i] = logfeature
    
    return tx

##################################### -- F LOG F-- ######################################

def transform_feature_log_feature(x, features):
    tx = x.copy()
    for i in features:
        feature = tx[:,i] + abs(np.min(tx[:,i])) + 0.1 #pour enlever les valeurs négatives
        logfeature = np.log(feature)
        for l in range(len(tx[:,i])):
                  tx[l,i] = tx[l,i] * logfeature[l]    
    
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

#####################################  --  TAN -- ###################################

def transform_feature_tan(x, features):
    tx = x.copy()
    for i in features:
        feature = tx[:,i]
        tanfeature = np.tan(feature)
        #add the new feature at the end !
        #c = np.c_[tx, cosfeature]
        #tx = c.copy()
        tx[:, i] = tanfeature
    
    return tx

#####################################  --  ARCSIN -- ###################################

def transform_feature_arcsin(x, features):
    tx = x.copy()
    for i in features:
        feature = tx[:,i]
        arcsinfeature = np.arcsin(feature)
        tx[:, i] = arcsinfeature
    
    return tx

#####################################  --  ARCCOS -- ###################################

def transform_feature_arccos(x, features):
    tx = x.copy()
    for i in features:
        feature = tx[:,i]
        arccosfeature = np.arccos(feature)
        tx[:, i] = arccosfeature
    
    return tx

#####################################  --  ARCTAN -- ###################################

def transform_feature_arctan(x, features):
    tx = x.copy()
    for i in features:
        feature = tx[:,i]
        arctanfeature = np.arctan(feature)
        tx[:, i] = arctanfeature
    
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
    
#####################################  --  new features for PRI_jet_num -- ###################################

def new_feature_PRI_jet_num(x, value):
    
    (l, c) = np.shape(x)
    tx = x.copy()
    
    #creat new vectors full of zeros
    tx0 = np.empty((l,), int)
    
    for i in range(l):
        if(tx[i,22] == value):
            #tx0 = np.append(tx0, 1, axis=0)
            tx0[i] = 1
        else:
            #tx0 = np.append(tx0, 0, axis=0)
            tx0[i] = 0

   
    c = np.c_[tx, tx0]
    tx = c.copy()     
    
    return tx
    
