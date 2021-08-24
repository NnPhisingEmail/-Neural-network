# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 17:05:58 2021

@author: Peleg
"""
from keras import backend as K

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS =  48531  #4000 used to be but 48531 is the unique words
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 1200
# This is fixed.
EMBEDDING_DIM = 300

# Split to Train and Test
test_size = 0.1
random_state = 42
#########################################
#         Pytorch Model - LSTM          #
#########################################
#  This version performs the same function as Dropout, however, it drops entire 1D feature maps instead of individual elements.
SpatialDropout1D = 0.1 # 20% dropout -- 2:25 CHANGED

#model parameters
neurons = 300 # dimensionality of the output space (300 Neurons)
dropout = 0.1 # Fraction of the units to drop for the linear transformation of the inputs
recurrent_dropout = 0.2 # Fraction of the units to drop for the linear transformation of the recurrent state

# complie model with: 
loss_function = 'categorical_crossentropy'
optimizer = 'adam'                            

# Dense Layer:
number_of_classes = 4 # 4 Classes at Dense layer
activation = 'softmax'

# Number of epochs (num of iterations to run all the data samples , Pack iteration with batch size of the data) all_data / batch_size 
epochs = 3000
batch_size = 25

# model fit:
validation_split = 0.1
patience=3
min_delta = 0.0001
monitor='val_loss'

hist_recall = []
hist_precision = []
hist_f1 = []

def recall_m(y_true, y_pred):
    global hist_recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    hist_recall.append(recall)
    return recall

def precision_m(y_true, y_pred):
    global hist_precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    hist_precision.append(precision)
    return precision

def f1_m(y_true, y_pred):
    global hist_f1
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    f1 = 2*((precision*recall)/(precision+recall+K.epsilon()))
    hist_f1.append(f1)
    return f1