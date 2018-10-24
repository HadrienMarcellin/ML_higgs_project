# -*- coding: utf-8 -*-
"""machine learning functions for project 1."""
import numpy as np


def create_cross_validation_datasets(N, nb_sets):
    
    test_percentage = 1/nb_sets
    test_length = int(round(N*test_percentage))
    train_length = int(N-test_length)
    
    random_indices = np.random.permutation(N)
    
    train_indices = np.zeros((train_length, nb_sets))
    test_indices = np.zeros((test_length, nb_sets))
    
    for i in range(nb_sets):
        train_indices[:,i] = np.delete(random_indices, list(range(i * test_length, (i + 1) * test_length)))
        test_indices[:,i] = random_indices[i * test_length : (i+1)*test_length]

    return test_indices, train_indices


##################################### -- TRANSFORM NAN values to 0 -- ##################

def transform_nan_to_zero(x):
    """   
    Parameters 
    -----------
    tx : np.array
        N x D matrix with features' values. Rows are samples and coluns are features.
    make_copy : bool
        If True, make a copy of tx before computation, False will overwrite
   
    Returns
    ----------
    tx_nan : np.array
        N x D matrix, with same size as input with 0 values instead of np.nan.
    """
    tx = x.copy()  
    tx[np.isnan(tx)] = 0
    
    return tx


##################################### -- TRANSFORM NAN values to mean -- ##################
#def transform_to_mean(x):
    
    #t = x.copy()
    #(l,c) = np.shape(t)
    #mean=np.zeros(c)
    
    ##calcul moyenne des features
    #for i in range(c):
        #somme = 0
        #sum_samples = 0
        #for j in range(l):
            #if(t[j,i] != np.nan):
                #somme = somme + t[j,i]
                #sum_samples = sum_samples + 1
        #t = x.copy()
    #(l,c) = np.shape(t)
    #mean=np.zeros(c)
    
    ##calcul moyenne des features
    #for i in range(c):
        #somme = 0
        #sum_samples = 0
        #for j in range(l):
            #if(t[j,i] != np.nan):
                #somme = somme + t[j,i]
                #sum_samples = sum_samples + 1
        
        #moyenne = somme/sum_samples
        #mean[i] = moyenne

    ##parcourt les nan et les mets à la moyenne de la feature
    #for i in range(c):
        #for j in range(l):
            #if(t[j,i] == np.nan):
                #t[j,i] = mean[i]
        #moyenne = somme/sum_samples
        #mean[i] = moyenne

    ##parcourt les nan et les mets à la moyenne de la feature
    #for i in range(c):
        #for j in range(l):
            #if(t[j,i] == np.nan):
                #t[j,i] = mean[i]
    

    #return t, mean

def transform_to_mean(x):

    t = x.copy()
    
    mean = np.nanmean(t, axis = 0)
    
    for column in t.T:
        column[np.isnan(column)] = np.nanmean(column)
    
    return t, mean

##################################### --  TRANSFORM TEST SET NAN TO PRE-COMPUTED MEAN -- ##################
def transform_to_precomputed_mean(x, mean):

    t = x.copy()
       
    for i, column in enumerate(t.T):
        column[np.isnan(column)] = mean[i]
    
    return t
    
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

#####################################  --  log -- ###################################
#mettre en log les features qui ont une distribution enxponentielle
def log(feature):
    logfeature = np.log(feature)
    return logfeature


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
    mean_x = np.nanmean(x, axis = 0)
    tx = tx - mean_x
    std_x = np.nanstd(x, axis = 0)
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


##################################### -- ADD BOOL WHEN NAN -- #######################

def add_bool_when_nan(tx, feature_ids):
    
    x = tx.copy()
    
    for id_ in feature_ids:
        bool_vect = np.zeros(len(x[:, id_]))
        bool_vect[np.isnan(x[:, id_ ])] = 1
        x = np.hstack( (x, bool_vect.reshape(len(bool_vect), 1)) )
    
    return x
    



#####################################  --  MSE -- ###################################

def calculate_mse(e):
    return 0.5*np.mean(e**2)


#####################################  --  MAE -- ###################################

def calculate_mae(e):
    return np.mean(np.abs(e))

#####################################  --  LOSS -- ###################################

def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    return calculate_mse(e)


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
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              #bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
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

        #print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              #bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws


#####################################  --  BATCH ITER  -- ###################################

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

#####################################  --  LEAST SQUARE  -- ###################################

def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)




#####################################  --  RIDGE REGRESSION  -- ###################################

def ridge_regression(y, tx, lamb):
    aI = lamb * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)



#####################################  --  Logistic FUNCTION  -- ###################################

def logistic_fun(predictioni):
    return 1 / (1 + np.exp(-predictioni))


#####################################  --  Calculate logistic cost function  -- ###################################

def logistic_cost(y, X, w):
    loss_log = 0
    for i in range(X.shape[0]):
        loss_log = loss_log + np.log(1+np.exp(X[i,:].T@w)) - y[i,] * X[i,:].T@w
    return loss_log
    

#####################################  --  Logistic Gradient -- ###################################

def logistic_gradient(y, X, w):
    return X.T@(logistic_fun(X@w)-y)
    

#####################################  --  Logistic Hessian Matrix -- ###################################

def logistic_hessian(X, w):
    S=np.zeros((X.shape[0],X.shape[0]))
    for i in range(X.shape[0]):
        S[i,i]=logistic_fun(X[i,:].T@w)*(1-logistic_fun(X[i,:].T@w))
    return X.T@S@X
    

#####################################  --  Logistic GRADIENT DESCENT -- ###################################

def log_gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad = logistic_gradient(y, tx, w)
        loss = logistic_cost(y, tx, w)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              #bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws
#PROBLEM WITH THE INVERSION OF MATRIX.

#####################################  --  Logistic Newton Method -- ###################################

def log_Newton_method(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        H=logistic_hessian(tx, w)
        grad = logistic_gradient(y, tx, w)
        loss = logistic_cost(y, tx, w)
        # gradient w by descent update
        H_inv=np.linalg.inv(H)
        w = w - gamma * H_inv@grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("1\n")
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              #bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

#####################################  --  Log_likelihood  -- ###################################
"""
def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return ll

#####################################  --  Logistic Regression Gradient ascent  -- ###################################

def logistic_regression(features, target, num_steps, learning_rate, add_intercept = False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))
        
    weights = np.zeros(features.shape[1])
    
    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient
        
        # Print log-likelihood every so often
        #if step % 10000 == 0:
            #print log_likelihood(features, target, weights)
        
    return weights
"""
#####################################  --  Optimal weight-vector of the return tuple -- ###################################
"""
    Return the optimal weights vector according to the minimal loss. 
    Useless for actual logistic regression.
    
    Parameters
    ----------
    This function take the returns items of the regression (vector and tuple)
    
    loss_reg : loss vector after regression
    
    w_reg : weight tuple after regression
    
    
    Returns
    -------
    loss_min == int value
    w_optimal  == optimal weight vector (1*D)
    
"""
    
def minimum_loss_vector(loss_reg,w_reg):
    loss_min=np.nanmin(loss_reg)
    i=np.where(loss_reg==loss_min)[0] 
    w_optimal = w_reg[i[0]]
    return loss_min, w_optimal
    
#####################################  --  Accuracy calculator -- ###################################
"""
    Return the accuracy of the model according to a labelized test-set.
    
    Parameters
    ----------
    
    y_pred = y vector predicted from the regression
    
    y_true = actual true vector of y
    
    
    Returns
    -------
    The accuracy of the model regarding this set
"""

def accuracy_calculator(y_pred, y_true):
    compt=0
    for i in range(len(y_pred)):
        if y_pred[i]==y_true[i]:
            compt=compt+1
    return compt/len(y_pred)
    
#####################################  --  SPLIT DATA -- ###################################
    
def split_data(x, y, ratio, myseed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(myseed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te


