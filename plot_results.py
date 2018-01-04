# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 08:53:24 2017

@author: Gerardo Cervantes
"""

#Used to plots graphs in python
import matplotlib.pyplot as plt


#Gini and Gini_val are a list of the gini's coefficeint at every epoch
#Gini_val is Gini's Coefficient on the test set
def plot_gini_results(gini, gini_val):
    plt.plot(gini)
    plt.plot(gini_val)
    
    plt.title('Model Gini coefficient')
    plt.ylabel('Gini coefficient')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    
    plt.show()