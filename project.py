# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:37:54 2017

@author: Gerardo Cervantes
"""

#This script creates a neural network for the Porto Seguro Kaggle competition
#https://www.kaggle.com/c/porto-seguro-safe-driver-prediction
#Predictions are generated and saved into a csv output file
#Gini's coefficient is the metric that the competition uses for scoring

import numpy as np

#Keras Library for neural networks
from keras.models import Sequential
from keras.layers import Dense, Dropout, PReLU
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras import regularizers, optimizers

#Used for AUC loss function
import tensorflow as tf

from read_csv_file import export_csv_file
from roc_auc_callback import roc_auc_callback
from generate_features import generate_features
from preprocess_data import preprocess
from plot_results import plot_gini_results

from sklearn.model_selection import train_test_split

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #uncomment to use CPU version of tensorflow

#Trains the neural network, Gini's coefficient for train and test is shown every epoch.
#Returns a gini_callback, which contains a list of Gini's Coefficient on train and test.
def train_neural_network(model, batchSize, nEpochs, x_train, y_train, x_test, y_test):
        
    gini_callback = roc_auc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))
    
    model.fit(x=x_train, y=y_train, batch_size=batchSize, epochs=nEpochs, verbose=2, callbacks=[gini_callback], validation_data= (x_test,y_test))
    
    return gini_callback
    
#AUC loss function - Area under curve loss function which more closley resembles Gini's coefficient, as relative ordering matters.
def pair_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    parts = tf.dynamic_partition(y_pred, y_true, 2)
    y_pos = parts[1]
    y_neg = parts[0]
    y_pos = tf.expand_dims(y_pos, 0)
    y_neg = tf.expand_dims(y_neg, -1)
    out = K.sigmoid((y_neg - y_pos)*10) #Change
    return K.mean(out)

#Adds the layers to the neural network
def add_neural_network_layers(model, numberInputNodes, numberOutputNodes):
    reg = 0.01 #regularization
    #Input layer
    model.add(Dense(numberInputNodes, input_shape=(numberInputNodes,), activation = 'relu'))
    
    model.add(Dense(512,  kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg))) #512
    model.add(PReLU())
    model.add(Dropout(0.8)) #0.8
    model.add(BatchNormalization())
    model.add(Dense(64, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg))) #64
    model.add(PReLU())
    model.add(Dropout(0.8))
    model.add(BatchNormalization())
    
  
    #Output layer
    model.add(Dense(numberOutputNodes, activation='sigmoid', kernel_initializer='he_uniform'))
    return model

#Creates neural network from x_train and y_train, uses x_validation and val_ids
#Creates neural network and trains it.
#Creates csv file to be used for the competition
def puertoPredictions(x_train, y_train, x_validation, validation_ids):
    
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size= 0.20) #Splits training and test data
    
    model = Sequential()
    add_neural_network_layers(model, len(x_train[0]), 1) 
    model.compile(loss='binary_crossentropy', metrics = ['binary_crossentropy'], optimizer=optimizers.Nadam(lr=0.00011))
#    model.compile(loss=pair_loss, metrics = [pair_loss], optimizer=optimizers.Nadam(lr=0.00008)) #Using AUC loss function
    batchSize = 64
    
    nEpochs = 20
    
    print('Training!')
    
    roc_auc = train_neural_network(model, batchSize, nEpochs, x_train, y_train, x_test = x_test, y_test = y_test)
    plot_gini_results(roc_auc.gini, roc_auc.gini_val)
    nn_output = model.predict(x_validation).flatten()
    export_csv_file("output.csv", validation_ids, nn_output)

    
x_train_file_name, x_validate_file_name = 'Data/train.csv', 'Data/test.csv'

x_train, y_train, x_validation, validation_ids = generate_features(x_train_file_name, x_validate_file_name);
    
x_train = np.array(x_train).astype(np.float32)
y_train = np.array(y_train).astype(np.int32)
x_validation = np.array(x_validation).astype(np.float32)
validation_ids = np.array(validation_ids).astype(np.int32)


x_train, y_train, x_validation = preprocess(x_train, y_train, x_validation)

puertoPredictions(x_train, y_train, x_validation, validation_ids)