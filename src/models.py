# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 22:12:37 2018

@author: pzq
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import time

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb



clf_rf = RandomForestClassifier(n_estimators=10)

def custom_eval( y_pred, d_matrix):
    belta = 2
    metric_name = "f2_score"
    label = d_matrix.get_label()
    # f2_score
    y_pred_d = (y_pred>=0.5)*1
    tp = np.sum(label[np.where(y_pred_d == 1)] == 1)
    fp = np.sum(label[np.where(y_pred_d == 1)] == 0)
    fn = np.sum(label[np.where(y_pred_d == 0)] == 1)
    #tn = np.sum(label[np.where(y_pred_d == 0)] == 0)
    
    precise = tp/(tp+fp) if not (tp+fp)==0 else 0
    recall  = tp/(tp+fn) if not (tp+fn)==0 else 0
    f_belta = (1+belta**2)*recall*precise/(belta**2*precise+recall) if not precise*recall==0 else 0
    
    # return metric_name 
    return metric_name, f_belta


def custom_loss( y_pred, d_matrix):
    penalty = 10.0
    #metric_name = "loss_penalty_"+str(int(penalty))
    label = d_matrix.get_label()
    # grad hess
    grad = -penalty*label/y_pred+(1-label)/(1.000001-y_pred)
    hess = penalty*label/(y_pred**2)+(1-label)/((1.000001-y_pred)**2)
    # return metric_name 
    return grad, hess


class CLF_XGB():
    def __init__(self, X_train, y_train,X_test, y_test):
        # init model
        # data type transform
        self.X_test = X_test
        self.dtrain = xgb.DMatrix(X_train, label = y_train)
        self.dtest = xgb.DMatrix(X_test, label = y_test)

    def load_model(self, model_name):
        pass
        
    def train_model(self, params, rounds):
        # train model
        
        
        # set parameters
        self.params = params
        self.rounds = rounds
        self.watchlist = [(self.dtrain, 'train'),(self.dtest,'test')]
        
        self.model = xgb.train(self.params, self.dtrain, 
                          num_boost_round = self.rounds,
                          evals = self.watchlist,
                          #obj = custom_loss,
                          feval = custom_eval,
                          maximize=False
                          )
        self.__save_model()
        #self.get_feature_scores()
    
    def __save_model(self):
        # save our model
        model_path = "model"
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_time = time.localtime(time.time())
        model_name = "xgb_%d_%d_%d_%d.model"%(model_time.tm_mon,
                                              model_time.tm_mday,
                                              model_time.tm_hour,
                                              model_time.tm_min)
        #self.model.save_model(os.path.join(model_path,model_name))
    
    def predict(self, X_test = None, threshold = 0.5):
        if not type(X_test) == type(None):
            dtest = xgb.DMatrix(X_test)
        else :
            #dtest = self.dtest
            dtest = xgb.DMatrix(self.X_test)
        self.y_pred_continuous = self.model.predict(dtest)
        #self.y_pred_continuous = 1.0 / (1.0 + np.exp(-self.y_pred_continuous))
        self.y_pred_discrete = (self.y_pred_continuous>= threshold)*1
        '''
        print("CLF_THRESHOLD = %.2f"%threshold)
        print("y_pred:")
        print("[0.0,0.1): %d"%((self.y_pred_continuous>=0.0)&(self.y_pred_continuous<0.1)).sum())
        print("[0.1,0.2): %d"%((self.y_pred_continuous>=0.1)&(self.y_pred_continuous<0.2)).sum())
        print("[0.2,0.3): %d"%((self.y_pred_continuous>=0.2)&(self.y_pred_continuous<0.3)).sum())
        print("[0.3,0.4): %d"%((self.y_pred_continuous>=0.3)&(self.y_pred_continuous<0.4)).sum())
        print("[0.4,0.5): %d"%((self.y_pred_continuous>=0.4)&(self.y_pred_continuous<0.5)).sum())
        print("[0.5,0.6): %d"%((self.y_pred_continuous>=0.5)&(self.y_pred_continuous<0.6)).sum())
        print("[0.6,0.7): %d"%((self.y_pred_continuous>=0.6)&(self.y_pred_continuous<0.7)).sum())
        print("[0.7,0.8): %d"%((self.y_pred_continuous>=0.7)&(self.y_pred_continuous<0.8)).sum())
        print("[0.8,0.9): %d"%((self.y_pred_continuous>=0.8)&(self.y_pred_continuous<0.9)).sum())
        print("[0.9,1.0]: %d"%((self.y_pred_continuous>=0.9)&(self.y_pred_continuous<=1.0)).sum())
        '''
        return self.y_pred_discrete
    
    def __create_feature_map(self, features):  
        outfile = open('xgb.fmap', 'w')  
        i = 0  
        for feat in features:  
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))  
            i = i + 1  
        outfile.close()  
        
    def get_feature_scores(self, features, i = 0):
        if not os.path.exists("csv"):
            os.mkdir("csv")
        if not os.path.exists("fig"):
            os.mkdir("fig")
        # get features
        self.features = features
        
        self.__create_feature_map(self.features)
        self.importance = self.model.get_fscore(fmap='xgb.fmap')
        self.importance = sorted(self.importance.items(), key=operator.itemgetter(1),
                                                     reverse = True)
        df = pd.DataFrame(self.importance, columns=['feature', 'fscore'])  
        df['fscore'] = df['fscore']/df['fscore'].sum()
        df.to_csv("csv/feat_importance_%2d.csv"%i, index=False) 
        
        #plt.figure()
        
        df = df.sort_values('fscore',ascending=True)
        df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 10),
                position = 1)  
        plt.title('XGBoost Feature Importance')  
        plt.xlabel('relative importance') 
        
        #xgb.plot_importance(self.model)
        plt.savefig("fig/feature_importance_%2d.png"%i)
        plt.close()
        #plt.show()
        
        # top 3 features
        self.top3_features = list(df['feature'][-3:])
        
        
        