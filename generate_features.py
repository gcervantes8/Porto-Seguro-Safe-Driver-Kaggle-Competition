# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 14:13:03 2017

@author: Gerardo Cervantes
"""

#Code resembles this kaggle kernel: https://www.kaggle.com/arpitajena/top-10-on-lb-0-287-lgbm-avg-of-kernel-outputs
#This code reads from the csv files does some preprocessing to the features.
#Categorical features are one-hot encoded
#Some useless features are removed
#Adds a new feature that contains amount of missing values in sample

import pandas as pd
import numpy as np

#Train file is a path to the train csv file, should end in .csv
#Test file is a path to the test csv file, should end in .csv
#Returns Tuple with train feature matrix and labels, test features and test feature ids
def generate_features(train_file, test_file):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    
    validation_ids = test['id'].values
    y_train = train['target'].values
    
    train = train.drop(['target','id'], axis = 1)
    test = test.drop(['id'], axis = 1)
    
    
    train = add_missing_value_feature(train)
    test = add_missing_value_feature(test)
    
    train, test = remove_calc_features(train, test)
    
    cat_features = [a for a in train.columns if a.endswith('cat')]
    
    train = one_hot_encode(train, cat_features)
    test = one_hot_encode(test, cat_features)
        
    
    return train.values, y_train, test.values, validation_ids


#Parameters original features and the categorical features
#One hot encodes the categorical features
#Returns the new features with one hot encoded categorical features concatenated
def one_hot_encode(feats, cat_features):
    for feature in cat_features:
    	temp = pd.get_dummies(pd.Series(feats[feature]))
    	feats = pd.concat([feats,temp],axis=1)
    	feats = feats.drop([feature],axis=1)
    return feats

#Removes features that start with calc.
#In the kaggle competition discussion it was said they had no correlation to target function.
#Slightly better results were gotten when removing them
def remove_calc_features(train, test):
    col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
    train = train.drop(col_to_drop, axis=1)
    test = test.drop(col_to_drop, axis=1)
    return train, test

#Adds a feature to the features which contains the amount of missing values in that data entry
#Returns the features concatenated with the new feature
def add_missing_value_feature(features):
    features['negative_one_vals'] = np.sum((features==-1).values, axis=1)
    return features