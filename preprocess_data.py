# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:41:02 2017

@author: Gerardo Cervantes
"""

import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle

#Preprocess the data to prepare it for the neural network.

#Higher level function to preprocess the data with various techniques.
def preprocess(x_train, y_train, x_validation):
   
#    x_train, x_validation = add_missing_column_features(x_train, x_validation)
#    x_train = change_missing_values_to_mean(x_train)
#    x_validation = change_missing_values_to_mean(x_validation)

#    n = 60
#    x_train, x_validation, mn, evals = pca(x_train, x_validation, n)
    
    x_train, x_validation = normalizeData(x_train, x_validation)
    
    return x_train, y_train, x_validation

#Normalizes the data so the mean is 0 and standard deviation is 1.
#Parameters is the data as numpy matrix of size (samples, feature)
#Normalizes x_train and x_val with the same normalization.
#Returns the normalized data.
def normalizeData(x_train, x_validation):

    x_train_len = len(x_train)
    
    normalized_x = preprocessing.normalize(np.concatenate((x_train, x_validation), axis = 0))
    x_train = normalized_x[:x_train_len,:]
    x_validation = normalized_x[x_train_len:,:]
    return x_train, x_validation

#x_matrix in a numpy matrix of size (samples, feature).
#Replaces missing values (values with -1) to the mean of that column
def change_missing_values_to_mean(x_matrix):
    x_matrix[x_matrix == -1] = np.nan #Replaces negative values with NaN, so we can take mean
    
    
    avg_features = np.nanmean(x_matrix, axis = 0).astype(np.float32) #Takes mean, ignores NaN values
    
    
    avg_features = np.resize(avg_features, (1,len(avg_features)))
    avg_features = np.repeat(avg_features, len(x_matrix), axis = 0)
    np.copyto(x_matrix, avg_features, where = np.isnan(x_matrix))
    return x_matrix

#Original data is imalanced with ratio of 1:26, with most not claiming insurance
#Balances data by duplicating data of those who claimed insurance.
#This technique can sometimes help, but can usually cause overfitting
def balance_data_upscaling(x, y, factor):
    train_zero_indices = np.where(y == 0)[0] #Returns indices where y_train == 1
    train_one_indices = np.where(y == 1)[0] #Returns indices where y_train == 1
    
    
    x_train_zero = x[train_zero_indices]
    y_train_zero = y[train_zero_indices]
    
    x_train_one = x[train_one_indices]
    y_train_one = y[train_one_indices]
    
    x_train_one = np.repeat(x_train_one, factor, axis = 0)
    y_train_one = np.repeat(y_train_one, factor, axis = 0)
    
    x,y = np.concatenate((x_train_zero, x_train_one), axis = 0), np.concatenate((y_train_zero, y_train_one), axis = 0)
    x, y = shuffle(x, y, random_state=0)
    return x, y
