# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 21:14:18 2018

@author: pzq
"""

import os

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

# data pre_process

def move_label_to_last_col(data_frame, label_name = "acc_now_delinq"):
    ''' move label col to the last col
    '''
    label = data_frame.pop(label_name)
    features = data_frame.columns
    data_frame.insert(data_frame.shape[1],label_name,label)
    return data_frame, features




def sub_sampling_and_SMOTE(X_train_ori, y_train_ori, 
                           sub_mode = "ON",
                           sub_frac = 0.5,
                           sm_mode = "ON",
                           sm_frac = 0.5):
    X_train_sub,y_train_sub = X_train_ori,y_train_ori
    # sub sampling on the neg samples
    if sub_mode == "ON":
        sub_dict = { 0: int(np.sum(y_train_ori==0)*sub_frac)}
        rus = RandomUnderSampler(ratio = sub_dict, 
                                 random_state = 0)
        X_train_sub, y_train_sub = rus.fit_sample(
                                    X_train_ori,y_train_ori)
    # SMOTE on the pos samples
    X_train,y_train = X_train_sub, y_train_sub
    if sm_mode == "ON":
        sm_dict = { 1 : int(np.sum(y_train_sub==0)*sm_frac)}
        sm_kind = ['regular', 'borderline1', 'borderline2', 'svm']
        sm = SMOTE(ratio = sm_dict, kind = sm_kind[0],
                   random_state = 0)
        X_train, y_train = sm.fit_sample(
                                    X_train_sub,y_train_sub)
    print("Before sub & sm:")
    print("X_train_ori shape =",str(X_train_ori.shape))
    print("After sub & sm:")
    print("NEG number = %d"%np.sum(y_train==0))    
    print("POS number = %d"%np.sum(y_train==1)) 
    print("NEG : POS = %.2f"%(np.sum(y_train==0)/np.sum(y_train==1)))
    print("X_train shape",str(X_train.shape))
    return X_train,y_train

def cross_validate():
    pass




def create_train_sample_file(n = 1000):
    data_path = "../data"
    train_csv = "train.csv"
    train_sample_csv = "train_sample_large.csv"
    data_train = pd.read_csv(
            os.path.join(data_path,train_csv),
            low_memory = False)
    data_train_sample = data_train.sample(n)
    data_train_sample.to_csv(
            os.path.join(data_path,train_sample_csv),
            index = False)

if __name__ == "__main__":
    create_train_sample_file(10000)