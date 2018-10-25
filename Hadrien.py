import numpy as np
import math
from proj1_helpers import *
import matplotlib.pyplot as plt


##################################################################################"
def change_y_boundaries(y):
    
    ym = y.copy()
    ym[ym < 0] = 0
    
    return ym

#################################################################################"""""

def display_feature_distribution(y, x, nb_bins):
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
    tx = x.copy()
    
    nb_col = 2
    nb_row = int(tx.shape[1]/2)
    
    fig, axs = plt.subplots(nb_row,nb_col, sharey=False, squeeze=False, tight_layout=False, figsize=(15, 60))

    for i, feature in enumerate(tx.T):
        col = i % 2
        row = int((i-col)/2)
        feature_true, feature_false, nb_nan = split_data_according_to_truth(feature, y)
        axs[row, col].hist([feature_true, feature_false], histtype = 'bar',bins=nb_bins, label=['Higgs', 'No Higgs'])
        axs[row, col].set_title("Feature n° {0} with {1}% NAN".format(i, round(nb_nan/len(feature)*100, 1)))
        axs[row, col].legend(prop={'size': 10})
                                
    plt.show()
        
    
def split_data_according_to_truth(x,y):
    """
    Attention, tx et y doivent être ranger dans le même ordre. Split vector x according to labels from y.

    Parameters
    ----------
    x : np.array
        1D vector of feature, from N different samples.
    y : np.array
        1D vector of labels. Labels are -1 (false) or +1 (true)

    Returns
    -------
    x_true: np.array
        1D vector with all values from label true.
    x_flase: np.array
        1D vector with all values from label false.
    nb_NAN: int
        Number of NAN values that has been discarded.

    """
    thresh = -100
    x_true = x[np.where(np.logical_and(y > 0, x > thresh))]
    x_false = x[np.where( np.logical_and(y < 0, x > thresh))]
    nb_nan = np.count_nonzero(np.isnan(x))
    
    return x_true, x_false, nb_nan


def NAN_values_overview(tx, nb_bins):
    
    """
    This function aims to display the distribution of NAN values per features and per samples. We consider NAN values as such when their value is under the given threshold.
    This is usefull to identify if some features/samples are relevant to keep or discard.
    
    Parameters
    ----------
    tx : np.array
        N x D matrix of features. Rows are samples and coluns are features.
    nb_bins : int
        Number of bins for histogram display.
        
    Returns
    -------
    """

    nb_nan_per_feature = np.count_nonzero(np.isnan(tx), axis=0)
    nb_nan_per_sample = np.count_nonzero(np.isnan(tx), axis=1)
    
    fig, axs = plt.subplots(1, 2, sharey=False, squeeze=True, tight_layout=False, figsize=(10, 5))
    axs[0].hist(nb_nan_per_sample, bins=nb_bins, histtype='bar')
    axs[0].set_xlabel("number of NAN values")
    axs[0].set_ylabel("number of samples")

    axs[1].hist(nb_nan_per_feature, bins=20, histtype='bar')
    axs[1].set_xlabel("number of NAN values")
    axs[1].set_ylabel("number of features")

    plt.show()
    
    
    
def NAN_values_overview_matrix(x, y):
    
    """
    This function aims to display the distribution of NAN values per features and per samples. We consider NAN values as such when their value is under the given threshold.
    This is usefull to identify if some features/samples are relevant to keep or discard.
    
    Parameters
    ----------
    tx : np.array
        N x D matrix of features. Rows are samples and coluns are features.
    nb_bins : int
        Number of bins for histogram display.
        
    Returns
    -------
    """
    
    tx = x.copy()
    indices = np.argsort(y)
    tx = tx[indices]
    
    tx = tx + 999
    
    fig, axs = plt.subplots(1, 1, figsize=(5,10))
    ax1 = axs

    ax1.spy(tx, aspect = 'auto')


    plt.show()