# -*- coding: utf-8 -*-
"""machine learning functions for project 1."""

import numpy as np
import matplotlib.pyplot as plt
from Hadrien import *
from ML_methods import *



################################### Gradient Descent Exploration #############################

def gradient_descent_exploration(y, tx, ratio, gammas, initial_w, max_iters):
    
    x_tr, x_te, y_tr, y_te = split_data(tx, y, ratio, myseed=1)
    
    losses_tr = []
    losses_te = []
    
    for gamma in gammas:
        
        losses_iter, ws_iter = gradient_descent(y_tr, x_tr, initial_w, max_iters, gamma)
        min_loss, min_ws = minimum_loss_vector(losses_iter, ws_iter)
        
        losses_tr.append(min_loss)
        losses_te.append(compute_loss(y_te, x_te, min_ws))
     
    train_test_errors_visualization(gammas, losses_tr, losses_te, 'GD')
    
    min_loss, min_gamma = minimum_loss_vector(losses_te, gammas)
    
    print("Gradient Descent, Loss : {0}, Lambda : {1}".format(round(min_loss, 3), min_gamma))
    
################################### Gradient Descent Exploration #############################

def stochastic_gradient_descent_exploration(y, tx, ratio, gammas, initial_w, batch_size, max_iters):
    
    x_tr, x_te, y_tr, y_te = split_data(tx, y, ratio, myseed=1)
    
    losses_tr = []
    losses_te = []
    
    for gamma in gammas:
        
        losses_iter, ws_iter = stochastic_gradient_descent(y_tr, x_tr, initial_w, batch_size, max_iters, gamma)
        min_loss, min_ws = minimum_loss_vector(losses_iter, ws_iter)
        
        losses_tr.append(min_loss)
        losses_te.append(compute_loss(y_te, x_te, min_ws))
     
    train_test_errors_visualization(gammas, losses_tr, losses_te, 'SGD')
    
    min_loss, min_gamma = minimum_loss_vector(losses_te, gammas)
    
    print("Stochastic Gradient Descent, Loss : {0}, Lambda : {1}".format(round(min_loss, 3), min_gamma))
    


################################### RIDGE REGRESSION Exploration #############################

def ridge_regression_exploration(y, tx, ratio, lambdas):
    
    losses_tr = []
    losses_te = []
    losses_val = []
    
    for lambda_ in lambdas:
        
        x_cross, x_val, y_cross, y_val = split_data(tx, y, ratio, myseed=1)
                                                    
        ind_te, ind_tr = create_cross_validation_datasets(len(y_cross), 4)
                                                    
        ws = []
        loss_tr = []
        loss_te = []
        
        for val in list(range(ind_te.shape[1])):
            
            y_tr = y_cross[ind_tr[:,val]]
            y_te = y_cross[ind_te[:,val]]
            
            x_tr = x_cross[ind_tr[:,val], :]
            x_te = x_cross[ind_te[:,val], :]
            
            #x_tr, _, _ = standardize_with_nan(x_tr)
            #x_te, _, _ = standardize_with_nan(x_te)
            
            ws.append(ridge_regression(y_tr, x_tr, lambda_))
            
            loss_tr.append(compute_loss(y_tr, x_tr, ws[-1]))
            loss_te.append(compute_loss(y_te, x_te, ws[-1]))
        
        ws_cross = np.mean(ws, axis = 0)
        
        losses_tr.append(np.mean(loss_tr))
        losses_te.append(np.mean(loss_te))
        losses_val.append(compute_loss(y_val, x_val, ws_cross))
         
    train_test_valid_errors_visualization(lambdas, losses_tr, losses_te, losses_val, 'ridge')
    
    min_loss, min_lambda = minimum_loss_vector(losses_val, lambdas)
    
    print("Ridge Resgression, Loss : {0}, Lambda : {1}".format(round(min_loss, 3), min_lambda))
    
    
       
    
    
#####################################################################################################"


def logistic_stochastic_gradient_descent_exploration(y, tx, ratio, gammas, batch, initial_w, max_iters):
    
    y = change_y_boundaries(y)

    x_tr, x_te, y_tr, y_te = split_data(tx, y, ratio, myseed=1)
    
    
    losses_tr = []
    losses_te = []
    
    for gamma in gammas:
        
        losses_iter, ws_iter = log_stochastic_gradient_descent(y_tr, x_tr, initial_w, batch, max_iters, gamma)
        min_loss, min_ws = minimum_loss_vector(losses_iter, ws_iter)
        
        losses_tr.append(min_loss)
        losses_te.append(logistic_cost(y_te, x_te, min_ws))
     
    train_test_errors_visualization(gammas, losses_tr, losses_te, 'LOG GD')
    
    min_loss, min_gamma = minimum_loss_vector(losses_te, gammas)
    
    print("Gradient Descent, Loss : {0}, Lambda : {1}".format(round(min_loss, 3), min_gamma))
    
def logistic_gradient_descent_exploration(y, tx, ratio, gammas, initial_w, max_iters):
    
    y = change_y_boundaries(y)

    x_tr, x_te, y_tr, y_te = split_data(tx, y, ratio, myseed=1)
    
    
    losses_tr = []
    losses_te = []
    
    for gamma in gammas:
        
        losses_iter, ws_iter = log_gradient_descent(y_tr, x_tr, initial_w, max_iters, gamma)
        min_loss, min_ws = minimum_loss_vector(losses_iter, ws_iter)
        
        losses_tr.append(min_loss)
        losses_te.append(logistic_cost(y_te, x_te, min_ws))
     
    train_test_errors_visualization(gammas, losses_tr, losses_te, 'LOG GD')
    
    min_loss, min_gamma = minimum_loss_vector(losses_te, gammas)
    
    print("Gradient Descent, Loss : {0}, Lambda : {1}".format(round(min_loss, 3), min_gamma))
    
    

    

    

################################ DISPLAY TEST & TRAIN ERROR ######################################
        
def train_test_errors_visualization(lambds, mse_tr, mse_te, method):
    """visualization the curves of mse_tr and mse_te."""
    plt.figure(figsize=(10,5))
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("Hyper-parameter")
    plt.ylabel("rmse")
    plt.title("Evolution of errors with {0}".format(method))
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("test_train_errors with {0}".format(method))
    
    
################################ DISPLAY TEST & TRAIN ERROR ######################################
        
def train_test_valid_errors_visualization(lambds, mse_tr, mse_te, mse_val, method):
    """visualization the curves of mse_tr and mse_te."""
    plt.figure(figsize=(10,5))
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    #plt.semilogx(lambds, mse_val, marker=".", color='g', label='validation error')

    plt.xlabel("Hyper-parameter")
    plt.ylabel("rmse")
    plt.title("Evolution of errors with {0}".format(method))
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("test_train_errors with {0}".format(method))
