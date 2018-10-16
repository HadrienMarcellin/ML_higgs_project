# -*- coding: utf-8 -*-
"""machine learning functions for project 1."""
import numpy as np

##################################### -- TRANSFORM -999 values to NAN -- ##################
def transform_to_nan(x, thresh):
    """   
    Parameters 
    -----------
    tx : np.array
        N x D matrix with features' values. Rows are samples and coluns are features.
    thresh : float
        value underwhich we consider the feature as NAN.
    make_copy : bool
        If True, make a copy of tx before computation, False will overwrite
   
    Returns
    ----------
    tx_nan : np.array
        N x D matrix, with same size as input with np.nan values instead of np.float.
    """
    tx = x.copy()  
    tx[tx < thresh] = np.nan
    
    return tx


#####################################  --  NORMALIZE -- ###################################

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis = 1)
    x = x - mean_x
    std_x = np.std(x, axis = 1)
    x = x / std_x
    return x, mean_x, std_x
        

def standardize_with_nan(x):
    """
    Standardize the data, removing the mean and dividing by the standard deviation. 
    If some values are under the threshold, they are not considered in the operation and left untouched.
    
    Parameters
    ----------
    tx : np.array
        N x D matrix with features' values. Rows are samples and coluns are features.
    make_copy : bool
        If True, make a copy of tx before computation, False will overwrite
    
    Returns
    -------
    tx_normalized : np.array
        N x D matrix, with same size as input with normalized features.
    """
    tx = x.copy()
    mean_x = np.nanmean(x)
    tx = tx - mean_x
    std_x = np.nanstd(x)
    tx = tx / std_x
    
    return tx, mean_x, std_x

##################################### -- Remove Missing Data -- #####################

def remove_missing_data(tx, thresh, sample_feature):
    
    n_sample = tx.shape[0]
    n_feature = tx.shape[1]
    
    tx_cleaned = tx.copy()
    tx_cleaned[tx_cleaned < thresh ] = np.nan
    
    rows_to_delete = []
    
    for i, sample in enumerate(tx_cleaned):
        if (np.count_nonzero(~np.isnan(sample)) < (1-sample_feature) * n_feature):
            rows_to_delete.append(i)
           
    
    tx_cleaned = np.delete(tx_cleaned, rows_to_delete, axis = 0)
    
    return tx_cleaned

#####################################  --  MSE -- ###################################

def calculate_mse(e):
    return 0.5*np.mean(e**2)


#####################################  --  MAE -- ###################################

def calculate_mae(e):
    return np.mean(np.abs(e))


#####################################  --  LEAST SQUARES -- ###################################

def least_squares(y, tx):
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    return np.linalg.solve(a,b)


#####################################  --  GRADIENT -- ###################################

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


#####################################  --  GRADIENT DESCENT -- ###################################

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mae(err)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws


#####################################  --  STOCHASTIC GRADIENT DESCENT  -- ###################################
# ACHTUNG : USED THE SAME GRADIENT CALCULATION AS FOR THE NORMAL GRADIENT DESCENT ....

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)

        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws


#####################################  --  RIDGE REGRESSION  -- ###################################

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    aI = lamb * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)





