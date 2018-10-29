# -*- coding: utf-8 -*-
"""machine learning functions for project 1."""
import numpy as np
from features_engineering import *
from ML_methods import *

def create_inter_product(tx, features_1, features_2):
    for i in range(len(features_2)):
        tx = transform_feature_inter_product(tx, [features_1[i], features_2[i]])
    return tx

def create_product_with_vector(tx, features, vector):
    
    if features == 'all':
        features = list(range(tx.shape[1]))
    
    for i in features:
        featureproduct = tx[:,i] * vector
        tx = np.c_[tx, featureproduct]
    return tx

def transform_column_22_and_delete(tx_nan):
    
    txx = tx_nan.copy()

    #ajoute des vecteurs booleens pour chaque valeur (0.0, 1.0, 2.0 ou 3.0) de la feature 22
    tx0 = new_feature_PRI_jet_num(txx, 0.0)
    tx1 = new_feature_PRI_jet_num(tx0, 1.0)
    tx2 = new_feature_PRI_jet_num(tx1, 2.0)
    tx3 = new_feature_PRI_jet_num(tx2, 3.0)

    #enlever la colonne de la feature 22 with "delete(matrice, indice, colonne = 1)"
    tx3_final = np.delete(tx3, 22, 1)

    return tx3_final.copy()

def add_first_feature_column(tx):
    
    (l,c) = np.shape(tx)
    a = np.ones((l,1), float)
    tx_plus = np.c_[a, tx]
    return tx_plus


def feature_22_to_matrix(tx, delete_feature = True):
    
    matrix_22 = np.zeros((tx.shape[0], 4))
    
    matrix_22[np.where(tx[:,22] == 0),1] = 1
    matrix_22[np.where(tx[:,22] == 1),1] = 1
    matrix_22[np.where(tx[:,22] == 2),2] = 1
    matrix_22[np.where(tx[:,22] == 3),3] = 1
    
    if delete_feature:
        tx = np.delete(tx, 22, axis = 1)
    
    return tx, matrix_22

def remove_constant_features(x, cols):
    return np.delete(x, cols, axis = 1)

def basic_features_process(tx, features_square, features_log, features_sin, features_cos, features_sqrt):
    
    replace = True
        
    tx_square = transform_feature_power(tx, features_square, 2, replace)

    tx_log, min_log = transform_feature_log(tx_square, features_log, replace)

    tx_angle_stand = standardize_angles(tx_log, features_sin)
    tx_sin = transform_feature_sin(tx_angle_stand, features_sin, replace)
    
    tx_angle_stand = standardize_angles(tx_sin, features_cos)
    tx_cos = transform_feature_cos(tx_angle_stand, features_cos, replace)
    
    tx_sqrt, min_sqrt = transform_feature_sqrt(tx_cos, features_sqrt, replace)

    return tx_sqrt, min_log, min_sqrt 

def transform_feature_to_mean_given_training(x, mean):
    
    t = x.copy()
        
    for i, column in enumerate(t.T):
        column[np.isnan(column)] = mean[i]
    
    return t

def transform_feature_log_given_training(x, features, min_log):
    
    tx = x.copy()
        
    for i, feature in enumerate(features):
        
        f_min = min_log[i]
        temp = tx[:,feature] - f_min + 0.1 #pour enlever les valeurs négatives
        logfeature = np.sqrt(temp)
        c = np.c_[tx, logfeature]
        tx = c.copy()
    
    return tx

def transform_feature_sqrt_given_training(x, features, min_sqrt):
    
    tx = x.copy()
        
    for i, feature in enumerate(features):
        
        f_min = min_sqrt[i]
        temp = tx[:,feature] - f_min + 0.1 #pour enlever les valeurs négatives
        sqrtfeature = np.sqrt(temp)
        c = np.c_[tx, sqrtfeature]
        tx = c.copy()
    
    return tx


def ridge_regression_cross_validation(tx, y, ratio_cross, nb_cross, lambda_):
        
    x_cross, x_val, y_cross, y_val = split_data(tx, y, ratio_cross, myseed=1)

    ind_te, ind_tr = create_cross_validation_datasets(len(y_cross), nb_cross)

    ws = []
    loss_tr = []
    loss_te = []

    for val in list(range(ind_te.shape[1])):
        
        y_tr = y_cross[ind_tr[:,val]]
        y_te = y_cross[ind_te[:,val]]

        x_tr = x_cross[ind_tr[:,val], :]
        x_te = x_cross[ind_te[:,val], :]

        ws.append(ridge_regression(y_tr, x_tr, lambda_))

        loss_tr.append(compute_loss(y_tr, x_tr, ws[-1]))
        loss_te.append(compute_loss(y_te, x_te, ws[-1]))
        
        y_pred = predict_labels(ws[-1], x_te)
        print("Accuracy : {0}; Loss {1}".format(round(accuracy_calculator(y_pred, y_te), 4), round(loss_tr[-1], 4)))


    ws_cross = np.mean(ws, axis = 0)
    loss_val = compute_loss(y_val, x_val, ws_cross)
    
    y_pred = predict_labels(ws_cross, x_val)
    print("Cross-Validation, Accuracy : {0}; Loss {1}".format(round(accuracy_calculator(y_pred, y_val), 4), round(loss_val, 4)))
    
         
    return ws_cross
    