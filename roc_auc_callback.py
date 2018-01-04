# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 16:42:46 2017

@author: Gerardo Cervantes
"""

from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score


#This class allows us to see the Gini's Coefficient at every epoch
#Is a callback that is given to the Keras fit function.

class roc_auc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        

    def on_train_begin(self, logs={}):
        self.gini = []
        self.gini_val = []
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict_proba(self.x, verbose=0)
        roc = roc_auc_score(self.y, y_pred)
        logs['roc_auc'] = roc
        logs['norm_gini'] = ( roc * 2 ) - 1

        y_pred_val = self.model.predict_proba(self.x_val, verbose=0)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        logs['roc_auc_val'] = roc_val
        logs['norm_gini_val'] = ( roc_val * 2 ) - 1

        print('\rroc_auc: %s - roc_auc_val: %s - norm_gini: %s - norm_gini_val: %s' % (str(round(roc,5)),str(round(roc_val,5)),str(round((roc*2-1),5)),str(round((roc_val*2-1),5))), end=10*' '+'\n')
        self.gini.append(logs['norm_gini'])
        self.gini_val.append(logs['norm_gini_val'])
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        
        return
    
    