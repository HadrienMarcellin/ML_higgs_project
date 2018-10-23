# -*- coding: utf-8 -*-
"""machine learning functions for project 1."""

import numpy as np
import matplotlib.pyplot as plt
from ML_methods import split_data, gradient_descent, stochastic_gradient_descent, minimum_loss_vector, compute_loss, ridge_regression



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
    

def ridge_regression_exploration(y, tx, ratio, lambdas):
    
    x_tr, x_te, y_tr, y_te = split_data(tx, y, ratio, myseed=1)
    
    losses_tr = []
    losses_te = []
    
    for lambda_ in lambdas:
        
        ws = ridge_regression(y_tr, x_tr, lambda_)
        
        loss_tr = compute_loss(y_tr, x_tr, ws)
        loss_te = compute_loss(y_te, x_te, ws)
        
        losses_tr.append(loss_tr)
        losses_te.append(loss_te)
     
    train_test_errors_visualization(lambdas, losses_tr, losses_te, 'ridge')
    
    min_loss, min_lambda = minimum_loss_vector(losses_te, lambdas)
    
    print("Ridge Resgression, Loss : {0}, Lambda : {1}".format(round(min_loss, 3), min_lambda))
    
    
       
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
    
    