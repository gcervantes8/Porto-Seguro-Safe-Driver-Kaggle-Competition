# Porto-Seguro-Safe-Driver-Kaggle-Competition
Kaggle competition for predicting whether a driver will file an insurance claim in the next year.
https://www.kaggle.com/c/porto-seguro-safe-driver-prediction

## To run

Requirements:
* Python 3.6
* Tensorflow library (or any other backend library like Theano)
* Keras library
* Pandas library
* train.csv and test.csv files found on the kaggle competition website


## Data
train.csv is used for training our model.

test.csv is used to make predictions, we submit those predictions to kaggle for a score.

### Data Information

* We do not have much information about what each feature is because the features are anonymized.
* The data is heavily imbalanced for there is an imbalance of 26:1 with the majority of people not filing insurance claim.
* The data also has a lot of missing values represented as (-1) for any feature.
* There are 57 features per entry in the csv file..
* The missing values are contained in 13 features throughout the training set, of which only 4 are non-categorical features.

## Data Modifications/Preprocessing
We found there to be many categorical features that had to be converted into one-hot representations.  After converting categorical features to one-hot encoding, there was an increase in the number of features from 58 to 212 features.

We also normalized the data so that the mean would be 0 and the standard deviation would be 1.  This had little affect to our results.

## Gini's Coefficient
Normalized gini's coefficient is used as the metric in this competition.
A gini's coefficient of 0 means the predictions were random.
A gini's coefficient less than 0 means you performed worse than random.
A normalized gini's coefficient maximum score is 1.

We added a custom metric to our Neural Network model so we could see the Gini's Coefficient at every epoch.

## Neural Network model used.
We used Keras to create our Neural Network model.  For this competition we only used a neural network.

We added weight initalization of HE_Uniform and batch normalization, they did not improve our results, but they sped up the training time.

## Techniques used

### To handle data imbalance

#### Data upscaling

To handle the data imbalance we tried to upscale the data.  This duplicated the samples of our data that filed an insurance claim (We had much more sampled in the data that didn't file an insuranc claim).
This technique ultimately did not end up helping us as we got worse results.

#### AUC loss function (high batch size)

To handle the data imbalance we also tried a different loss function, we tried the AUC loss function.  We tried AUC loss function as this more closeley related the scoring metric was used for the competition.
For the Gini's coefficient scoring, only relative ordering matters, simlarly to the AUC loss function.

The results from AUC gave similar results to that of binary cross entropy but slightly worse results.  For AUC we were also using a high batch size so that there were samples from people who filed and insurance claim and people who didn't in the batch.


### To handle missing values

We initially replaced the missing values with the mean of that feature.  We later stopped doing this and found better results at leaving the values as -1 before normalization.  
We believe that the nerual network learned that those values are missing.


### Technique we couldn't use

We had the idea of augmenting the training data with some data from the testing data.  We would do this by treating the data as unlabeled data, and then augmenting the training data.
We found this to be against the competition rules, so we did not use it.

## Competition
There were 5,169 teams that entered the competition.

There was prize money of $25,000

# Best models used
* 2 hidden layers of size 512, and 64.
* Batch normalization after every hidden layer
* he uniform weight initalization
* Dropout of 0.8
* Regularization of 0.01
* Loss function: Binary Cross entropy
* Optimizer: Nadam


## Results
We used an ensemble technique for our final submission.

We learned that ensembling multiple models is usually helpful for Kaggle competitions like this, it helped improve our score.

Our final results for the competition was:

0.27427 for the public leaderboard

0.27970 for the private leaderboard

This put us at 3,158th place

These results were the best we got before the competition ended, they did not use the ensemble technique.

After the competition ended we got better results.

0.27668 for the public leaderboard

0.28085 for the private leaderboard


## Authors:
Gerardo Cervantes

Farid Garciayala

Jose R. Torres
