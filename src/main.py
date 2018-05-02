# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 20:07:48 2018

@author: pzq
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

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
PCA_MODE = "OFF"
PCA_DIMENTION = 15

test_size = 0.33


# during STEP 3
SUB_MODE = "OFF"
SM_MODE = "OFF"
sub_frac = 0.02
sm_frac = 1

CLF_THRESHOLD = 0.01


# during STEP 4
# model settings
params = {
        'booster':'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth':15,
        'lambda':3,
        'subsample':0.75,
        'colsample_bytree':0.75,
        'min_child_weight':1,
        'eta': 0.3,  # alias :learning rate
        
        'scale_pos_weight':85, # NEG : POS
        'max_delta_step':100,  # range [0, inf)
        
        'early_stopping_rounds':10,
        
        'seed':0,
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


# PCA analysis
pca = PCA(n_components = PCA_DIMENTION)
X_data_pca = None
which_X = 0
if PCA_MODE == "ON":
    X_data_pca = pca.fit_transform(X_data)
    which_X = 1


# split original data into train test sets 
X = (X_data, X_data_pca)
X_train_ori , X_test, y_train_ori , y_test = train_test_split(
        X[which_X],y_data, test_size = test_size, random_state = 1)




#=============================STEP 3============================
# do sub_sampling and SMOTE on train set
X_train,y_train = dtps.sub_sampling_and_SMOTE(X_train_ori, y_train_ori,
                                             sub_mode = SUB_MODE,
                                             sub_frac= sub_frac,
                                             sm_mode = SM_MODE,
                                             sm_frac = sm_frac
                                             )





#=============================STEP 4============================
# Random Forest (NOT GOOD)
#clf_rf.fit(X_train,y_train)
#y_pred = clf_rf.predict(X_test)


# XGBOOST 
clf_xgb = CLF_XGB(X_train, y_train, X_test, y_test )
clf_xgb.train_model( params, iter_num)
clf_xgb.get_feature_scores(features)
y_pred = clf_xgb.predict(threshold = CLF_THRESHOLD)
# visualize top 3 importance features
if PLT_MODE == "ON":
    plot_feature_3d(data_train,clf_xgb.top3_features, p=0)
    plt.show()


#=============================STEP 5============================
# evaluate our model on test set

result = Evaluate(y_test, y_pred)

# evaluate our model on train set
y_pred_train = clf_xgb.predict(X_test = X_train, threshold = CLF_THRESHOLD)
result = Evaluate(y_train, y_pred_train,mode="train")

#=============================STEP 6============================
# predict on test.csv and get our final result
'''
data_test = pd.read_csv(
        os.path.join(data_path,test_csv),
            low_memory = False)

#data_test = dtps.feature_extract[feature_extract_choose](data_test, mode=feature_cat_mode)

data_test = pp.pre_process(data_test)
X_real_test = data_test.iloc[:,:].values

y_real_pred = clf_xgb.predict(X_real_test)

print("pred 1:",np.sum(y_real_pred==1))
print("pred 0:",np.sum(y_real_pred==0))
'''
