import numpy as np
import math
from proj1_helpers import *
import matplotlib.pyplot as plt



def display_feature_distribution(y, tx, thresh, nb_bins):
    """
    This function aims to display the distribution of the features. For each feature, we have N samples with different values. 
    In order to have relevant graph distributions, we will get rid of the NAN values (Xij < -900). 
    y vector is used to seperate samples and viualize the distribution of the feature according to the labels.
    
    Parameters
    ----------
    y : np.array
        1D vector of labels. Labels are -1 (false) or +1 (true)
    tx : np.array
        N x D matrix of features. Rows are samples and coluns are features.
    nb_bins : int
        number of bins for histogram display.
    Returns
    -------
    Display Graphs
    """
    
    
    nb_col = 2
    nb_row = int(tx.shape[1]/2)
    
    fig, axs = plt.subplots(nb_row,nb_col, sharey=False, squeeze=False, tight_layout=False, figsize=(15, 60))

    for i, feature in enumerate(tx.T):
        col = i % 2
        row = int((i-col)/2)
        feature_true, feature_false, nb_nan = split_data_according_to_truth(feature, y, -900)
        axs[row, col].hist([feature_true, feature_false], histtype = 'bar',bins=nb_bins, label=['True', 'False'])
        axs[row, col].set_title("Feature n° {0} with {1}% NAN".format(i, round(nb_nan/len(feature)*100, 1)))
        axs[row, col].legend(prop={'size': 10})
                                
    plt.show()
        
    
def split_data_according_to_truth(x,y,thresh):
    """
    Attention, tx et y doivent être ranger dans le même ordre. Split vector x according to labels from y.

    Parameters
    ----------
    x : np.array
        1D vector of feature, from N different samples.
    y : np.array
        1D vector of labels. Labels are -1 (false) or +1 (true)
    thresh : float
        threshold under which we consider the value as NAN

    Returns
    -------
    x_true: np.array
        1D vector with all values from label true.
    x_flase: np.array
        1D vector with all values from label false.
    nb_NAN: int
        Number of NAN values that has been discarded.

    """
    
    x_true = x[np.where((y>0) & (x > thresh))]
    x_false = x[np.where((y<0) & (x > thresh))]
    nb_nan = len(x[np.where(x<-900)])
    
    return x_true, x_false, nb_nan


def NAN_values_overview(tx, thresh, nb_bins):
    
    """
    This function aims to display the distribution of NAN values per features and per samples. We consider NAN values as such when their value is under the given threshold.
    This is usefull to identify if some features/samples are relevant to keep or discard.
    
    Parameters
    ----------
    tx : np.array
        N x D matrix of features. Rows are samples and coluns are features.
    thresh : float
        Threshold under which we consider the value as NAN
    nb_bins : int
        Number of bins for histogram display.
        
    Returns
    -------
    """
    
    temp = tx.copy()
    temp[np.where(temp > thresh)] = 0
    temp[np.where(temp < thresh)] = 1
    nan_sum_per_feature = temp.sum(axis=0)
    nan_sum_per_sample = temp.sum(axis=1)

    fig, axs = plt.subplots(1, 2, sharey=False, squeeze=True, tight_layout=False, figsize=(10, 5))
    axs[0].hist(nan_sum_per_sample, bins=nb_bins, histtype='bar')
    axs[0].set_xlabel("number of NAN values")
    axs[0].set_ylabel("number of samples")

    axs[1].hist(nan_sum_per_feature, bins=20, histtype='bar')
    axs[1].set_xlabel("number of NAN values")
    axs[1].set_ylabel("number of features")

    plt.show()