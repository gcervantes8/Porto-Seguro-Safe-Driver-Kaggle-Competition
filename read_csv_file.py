# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:12:25 2017

@author: Gerardo Cervantes
"""
import csv
import numpy as np
import pandas as pd

#Reads train and test CSV file and returns a numpy 2D matrix.
#Returns the training features and labels
#Returns x_validation, features that will be predicted and submitted to kaggle, and the sample ids.
def read_data(x_train_csv, x_test_csv):
    csv_train_items = read_csv(x_train_csv)
    
    x_train = csv_train_items[1:,2:].astype(np.float) #Removes 2 left columns (target and id) and first row (names)
    y_train = csv_train_items[1:,1].astype(np.float) #Gets 2nd column and removes first row 
    
    csv_validation_items = read_csv(x_test_csv)
    
    x_validation = csv_validation_items[0:,1:].astype(np.float)
    validation_ids = csv_validation_items[0:,:1].astype(np.int32).flatten() #Flatten convert matrix of (n,1) to (n,) vector 

    
    return x_train, y_train, x_validation, validation_ids

#Takes the csv file name and returns the csv file as matrix
def read_csv(csv_file_name):
    return pd.read_csv(csv_file_name).as_matrix()

#Takes the csv file names that you want to ensemble as a list.
#Averages the results of all the CSV files, ensembles are very useful for kaggle competitions
#CSV files should be in the format for the competition
def ensemble_results(csv_files):
    
    avg = 0
    csv_file_count = 0
    for csv_file in csv_files:
        matrix = read_csv(csv_file)
        ids = matrix[:,0].astype(int)
        avg = matrix[:,1] + avg
        csv_file_count += 1
    avg = avg/csv_file_count
    export_csv_file('Ensemble_output.csv', ids, avg)
    
#Exports into a format that Porto Seguro uses for the kaggle competition
#Export_path is the path where it should save the file, should include .csv extension
#Validation ids are the ids that come with the validation features, each sample has an id
#nn_output is an array with the predictions
def export_csv_file(export_path, val_ids, nn_output):
    
    with open(export_path, "w", newline="\n", encoding="utf-8") as f:
        
        writer = csv.writer(f)
        writer.writerow(["id", "target"])
        for i, val_item in enumerate(nn_output):
            b = val_ids[i]
            arr = [b, val_item]
            writer.writerow(arr)
    