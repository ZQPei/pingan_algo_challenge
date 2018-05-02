# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 11:49:47 2018

@author: pzq
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from imblearn.ensemble import EasyEnsemble 

import pre_process as pp
import data_process as dtps
from models import clf_rf, CLF_XGB
from evaluate import Evaluate
from visualize import plot_feature_3d, plot_PCA_3D

#=============================SETTINGS============================
# PARAMETER SETTINTS
data_path = "../data"
train_csv = "train.csv"
train_sample_csv = "train_sample.csv"
test_csv = "test.csv"
label_name = "acc_now_delinq"


# during STEP 1
feature_drop_mode = "OFF"
feature_retain_list = [
        "mths_since_last_major_derog",
        "tot_cur_bal",
        "revol_util",
        "int_rate",
        "revol_bal",
        "total_rev_hi_lim",
        "annual_inc",
        "sub_grade",
        "total_rec_int",
        "dti",
        "total_acc",
        "total_pymnt",
        "total_rec_prncp",
        "installment",
        "tot_coll_amt",
        "loan_amnt",
        "out_prncp",
        "term",
        "out_prncp_inv",
        ]

normalization_mode = "OFF"

# during STEP 2

test_size = 0.2


# during STEP 3
SUB_MODE = "OFF"
SM_MODE = "OFF"
sub_frac = 0.4
sm_frac = 1

CLF_THRESHOLD = 0.5


# during STEP 4
# model settings
params = {
        'booster':'gbtree',
        'objective': 'binary:logistic',
        #'eval_metric': 'logloss',
        #'max_depth':15,
        #'lambda':1,
        #'subsample':1,
        #'colsample_bytree':1,
        #'min_child_weight':1,
        #'eta': 0.3,  # alias :learning rate
        
        #'scale_pos_weight':85, # NEG : POS
        #'max_delta_step':100,  # range [0, inf)
        
        'seed':42,
        'nthread':8,
        'silent':1
         }
iter_num = 150

# whether to show plots
PLT_MODE = "OFF"

#=============================STEP 1============================
# read data and extract feature 
# adjust the label col to the right
data_train = pd.read_csv(
            os.path.join(data_path,train_csv),
            low_memory = False)
#data_train = dtps.feature_extract[feature_extract_choose](data_train, mode=feature_cat_mode)
data_train = pp.pre_process(data_train)
# whether to drop some features
if feature_drop_mode == 'ON':
    # retain features of high importance
    data_train = pp.retain_features(data_train, feature_retain_list)
if normalization_mode == "ON":
    # normalize features
    data_train = pp.normalize_feature(data_train)
data_train, features = dtps.move_label_to_last_col(data_train)

# visualize features
#plot_feature_3d(data_train, feature_names, n = 1000, p = 4)




#=============================STEP 2============================
# PCA optional
# split original data into train test sets 
X_data = data_train.iloc[:,:-1].values
y_data = data_train.iloc[:,-1].values




# split original data into train test sets 
X_train_ori , X_test, y_train_ori , y_test = train_test_split(
        X_data,y_data, test_size = test_size, random_state = 42)







#===========================FOR TEST==================================
#=========================EASY ENSEMBLE===============================

n_subsets = 50
ee = EasyEnsemble(random_state=42, n_subsets=n_subsets)
X_train, y_train = ee.fit_sample(X_train_ori, y_train_ori)
print("Num of each sets: %d" %y_train.shape[1])

clf_xgbs = []
y_preds = np.zeros((n_subsets,y_test.size))

for i in range(n_subsets):
    print("Round %3d"%(i))
    X_train_i = X_train[i]
    y_train_i = y_train[i]
    clf_xgb_i = CLF_XGB(X_train_i, y_train_i, X_test, y_test)
    clf_xgbs.append(clf_xgb_i)
    clf_xgb_i.train_model(params, iter_num)
    #clf_xgb_i.get_feature_scores(features,i)
    y_pred_i = clf_xgb_i.predict(threshold = CLF_THRESHOLD)
    y_preds[i] = y_pred_i
    

y_pred_tmp = y_preds.sum(0)/y_preds.shape[0]
y_pred = (y_pred_tmp>=0.999)*1
result = Evaluate(y_test, y_pred)
    









