# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 09:07:06 2018

@author: pzq
"""

import numpy as np
from sklearn import metrics

class Evaluate():
    def __init__(self,y_true,y_pred, mode="test"):
        self.mode = mode
        self.belta = 2
        # y_true y_pred should be a vector
        self.y_true = y_true
        self.y_pred = y_pred
        self.__evaluate()
        self.__print()

    def __evaluate(self):
        self.total = self.y_true.size
        self.p = np.sum(self.y_pred==1)
        self.n = np.sum(self.y_pred==0)
        # calculate TP FP FN TN 
        self.tp = np.sum(self.y_true[np.where(self.y_pred == 1)] == 1)
        self.fp = np.sum(self.y_true[np.where(self.y_pred == 1)] == 0)
        self.fn = np.sum(self.y_true[np.where(self.y_pred == 0)] == 1)
        self.tn = np.sum(self.y_true[np.where(self.y_pred == 0)] == 0)
        # accuracy 
        self.accuracy = (self.tp+self.tn)/self.total
        # precise & recall
        if self.tp == 0:
            self.precise = 0
            self.recall = 0
        else:
            self.precise = self.tp/(self.tp+self.fp)
            self.recall  = self.tp/(self.tp+self.fn)
        # f_belta
        if self.precise == 0 and self.recall == 0:
            self.f_belta = 0
        else:
            self.f_belta = (1+self.belta**2)*self.recall*self.precise \
                            /(self.belta**2*self.precise+self.recall)
        
    def __print(self):
        print("On %d %s sets, with %d pos values, our model's performance:"
              %(self.total,self.mode, np.sum(self.y_true==1)))
        print("TP = %d, FP = %d, FN = %d, TN = %d"
              %(self.tp,self.fp,self.fn,self.tn))
        print("Accuracy = %.4f"%self.accuracy)
        print("F2 score = %.4f"%self.f_belta)
        
        print ('AUC: %.4f' % metrics.roc_auc_score(self.y_true,self.y_pred))
        print ('ACC: %.4f' % metrics.accuracy_score(self.y_true,self.y_pred))
        print ('Recall: %.4f' % metrics.recall_score(self.y_true,self.y_pred))
        print ('Precesion: %.4f' %metrics.precision_score(self.y_true,self.y_pred))
        print ('F1-score: %.4f' %metrics.f1_score(self.y_true,self.y_pred))
                 