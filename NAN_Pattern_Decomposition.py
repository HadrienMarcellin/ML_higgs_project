# -*- coding: utf-8 -*-
"""machine learning functions for project 1."""
import numpy as np


def decompose_dataset_according_to_NAN_pattern(tx, y, pcols = [0, 23, 28]):
    
    p1_ind_full = np.isnan(tx[:,pcols[0]])
    p2_ind_full = np.isnan(tx[:,pcols[1]])
    p3_ind_full = np.isnan(tx[:,pcols[2]])

    p1_ind = p1_ind_full

    p2_ind = p2_ind_full & ~p1_ind

    p3_ind = p3_ind_full & ~p1_ind & ~p2_ind

    p0_ind = (p3_ind | True) & ~p3_ind & ~p1_ind & ~p2_ind
    
    p0 = tx[p0_ind, :]
    p1 = tx[p1_ind, :]
    p2 = tx[p2_ind, :]
    p3 = tx[p3_ind, :]
    
    y0 = y[p0_ind]
    y1 = y[p1_ind]
    y2 = y[p2_ind]
    y3 = y[p3_ind]
    
    #delete useless columns for each group regarding NaN values:
    # !!! need to decrement the index of our table for each new iteration 
    
    #p1=np.delete(p1,0,1)
    
    #p2=np.delete(p2,5,1)
    #p2=np.delete(p2,5,1)
    #p2=np.delete(p2,10,1)
    #p2=np.delete(p2,23,1)
    #p2=np.delete(p2,23,1)
    #p2=np.delete(p2,23,1)
    
    #p3=np.delete(p3,23,1)
    #p3=np.delete(p3,23,1)
    #p3=np.delete(p3,23,1)
    
    return p0_ind, p0, y0, p1_ind, p1, y1, p2_ind, p2, y2, p3_ind, p3, y3
