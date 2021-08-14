# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 17:05:58 2021

@author: Peleg
"""

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 4000
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
SpatialDropout1D = 0.2 # 20% dropout

#model parameters
neurons = 300 # dimensionality of the output space (300 Neurons)
dropout = 0.2 # Fraction of the units to drop for the linear transformation of the inputs
recurrent_dropout = 0.2 # Fraction of the units to drop for the linear transformation of the recurrent state

# complie model with: 
loss_function = 'categorical_crossentropy'
optimizer = 'adam'                            

# Dense Layer:
number_of_classes = 4 # 4 Classes at Dense layer
activation = 'softmax'

# Number of epochs (num of iterations to run all the data samples , Pack iteration with batch size of the data) all_data / batch_size 
epochs = 20
batch_size = 1

# model fit:
validation_split = 0.1
patience=3
min_delta = 0.0001
monitor='val_loss'
