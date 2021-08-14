# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 20:05:31 2021

@author: Peleg
"""


#########################################
#               TO-DO                 #
#########################################

#from nltk.corpus import stopwords
#from nltk import word_tokenize
#STOPWORDS = set(stopwords.words('english'))


#########################################
#               Imports                 #
#########################################
'''

'''
import lstm_parameters as param

'''
NumPy is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object,
various derived objects (such as masked arrays and matrices), and an assortment of routines for
fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier
transforms, basic linear algebra, basic statistical operations, random simulation and much more
'''
import numpy as np 

'''
Pandas is an open source Python package that is most widely used for data science/data analysis and machine learning tasks.
It is built on top of another package named Numpy, which provides support for multi-dimensional arrays.
'''
import pandas as pd

'''
matplotlib. pyplot is a collection of functions that make matplotlib work like MATLAB.
Each pyplot function makes some change to a figure: e.g., creates a figure, creates a plotting area in a figure,
plots some lines in a plotting area, decorates the plot with labels, etc.
'''
import matplotlib.pyplot as plt

'''
This class allows to vectorize a text corpus,
by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary) or 
into a vector where the coefficient for each token could be binary, based on word count, based on tf-idf..
By default, all punctuation is removed, turning the texts into space-separated sequences of words (words maybe include the ' character).
These sequences are then split into lists of tokens. 
They will then be indexed or vectorized.
'''
from keras.preprocessing.text import Tokenizer

'''
pad_sequences is used to ensure that all sequences in a list have the same length.
By default this is done by padding 0 in the beginning of each sequence until each sequence has the same length as the longest sequence.
'''
from keras.preprocessing.sequence import pad_sequences

'''
lets you create a model layer by layer for most problems.
It’s straightforward (just a simple list of layers), but it’s limited to single-input, single-output stacks of layers.
'''
from keras.models import Sequential

'''
Layers are the basic building blocks of neural networks in Keras.
A layer consists of a tensor-in tensor-out computation function (the layer's call method) and some state, held in TensorFlow variables (the layer's weights).

Core layers used: Dense layer, Embedding layer.
#################################################
#               Dense Layer                     #
#################################################
link: https://keras.io/api/layers/core_layers/dense/

The dense layer is a neural network layer that is connected deeply, which means each neuron in the dense layer receives input from all neurons of its previous layer. 
The dense layer is found to be the most commonly used layer in the models. ... Thus, dense layer is basically used for changing the dimensions of the vector.

#################################################
#               Embedding Layer                 #
#################################################
link: https://keras.io/api/layers/core_layers/embedding/
The Embedding layer is defined as the first hidden layer of a network. ... input_length: This is the length of input sequences, 
as you would define for any input layer of a Keras model. For example, if all of your input documents are comprised of 1000 words, this would be 1000.


#################################################
#Recurrent layer: Lstm - long short term memory #
#################################################
link: https://keras.io/api/layers/recurrent_layers/lstm/

Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of learning order dependence in sequence prediction problems. 
This is a behavior required in complex problem domains like machine translation, speech recognition, and more. 
LSTMs are a complex area of deep learning.

#################################################
#   Regularization layers: SpatialDropout1D     #
#################################################
link: https://keras.io/api/layers/regularization_layers/spatial_dropout1d/

Spatial 1D version of Dropout.

This version performs the same function as Dropout, however, it drops entire 1D feature maps instead of individual elements.
If adjacent frames within feature maps are strongly correlated (as is normally the case in early convolution layers) then
regular dropout will not regularize the activations and will otherwise just result in an effective learning rate decrease.
In this case, SpatialDropout1D will help promote independence between feature maps and should be used instead.
#################################################
'''
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

'''
#################################################
#               train_test_split                #
#################################################
is a function in Sklearn model selection for splitting data arrays into two subsets:
for training data and for testing data.
With this function, you don't need to divide the dataset manually.
By default, Sklearn train_test_split will make random partitions for the two subsets.
'''
from sklearn.model_selection import train_test_split

'''
#################################################
#               keras.callbacks                 #
#################################################
EarlyStopping:

Keras supports the early stopping of training via a callback called EarlyStopping.
This callback allows you to specify the performance measure to monitor, the trigger, and once triggered, it will stop the training process.
The EarlyStopping callback is configured when instantiated via arguments.

#################################################
ModelCheckpoint:

The ModelCheckpoint callback class allows you to define where to checkpoint the model weights, 
how the file should named and under what circumstances to make a checkpoint of the model.
The API allows you to specify which metric to monitor, such as loss or accuracy on the training or validation dataset.
#################################################
'''
from keras.callbacks import EarlyStopping,ModelCheckpoint

'''
Cufflinks is another library that connects the Pandas data frame with Plotly enabling users to create visualizations directly from Pandas.
The library binds the power of Plotly with the flexibility of Pandas for easy plotting.
'''
import cufflinks

'''
The Python interpreter can be used from an interactive shell.
The interactive shell is also interactive in the way that it stands between the commands or actions and their execution.
In other words, the shell waits for commands from the user, which it executes and returns the result of the execution.
'''
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

'''
#################################################
#               confusion_matrix                #
#################################################
sklearn.metrics.confusion_matrix(y_true, y_pred, labels=None)[source] Compute confusion matrix to evaluate the accuracy of a classification. 
By definition a confusion matrix is such that is equal to the number of observations known to be in group but predicted to be in group .

#################################################
#             ConfusionMatrixDisplay            #
#################################################
Confusion Matrix visualization.
'''
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

'''
Keras is a model-level library, providing high-level building blocks for developing deep learning models. 
It does not handle itself low-level operations such as tensor products, convolutions and so on. 
Instead, it relies on a specialized, well-optimized tensor manipulation library to do so, serving as the "backend engine" of Keras. 
Rather than picking one single tensor library and making the implementation of Keras tied to that library, Keras handles the problem in a modular way, 
and several different backend engines can be plugged seamlessly into Keras.
'''
from keras import backend as K

'''
In Python, date and time are not a data type of its own, but a module named datetime can be imported to work with the date as well as time. 
Datetime module comes built into Python, so there is no need to install it externally. 
Datetime module supplies classes to work with date and time. 
These classes provide a number of functions to deal with dates, times and time intervals. 
Date and datetime are an object in Python, so when you manipulate them, you are actually manipulating objects and not string or timestamps. 
'''
import datetime

'''
The OS module in Python provides functions for interacting with the operating system. 
OS comes under Python’s standard utility modules. 
This module provides a portable way of using operating system dependent functionality. 
The *os* and *os.path* modules include many functions to interact with the file system.
'''
import os

#########################################
#               Dataset                 #
#########################################
df = pd.read_csv('final_dataset.csv')
df = df.drop(columns=['Unnamed: 0'])
df_label_count = df.Label.value_counts()
df_label_count_sorted = sorted(df_label_count.index.values, key=lambda s: s.lower())  # creates label list by alphabetical order
print('Label Count:\n{}'.format(df_label_count))


#########################################
#          Global Metric Lists          #
#########################################

hist_recall = []
hist_precision = []
hist_f1 = []

#########################################
#     Checkpoint Folder and Saving      #
#########################################

checkpoint_path = './CNN/checkpoints/'
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
    
    
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    monitor='val_loss',
    mode='max',
    save_best_only=True)

#########################################
#             Functions                 #
#########################################
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

#########################################
#            Data Extraction            #
#########################################

tokenizer = Tokenizer(num_words=param.MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Email'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


X = tokenizer.texts_to_sequences(df['Email'].values)
X = pad_sequences(X, maxlen=param.MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(df['Label']).values
print('Shape of label tensor:', Y.shape)


X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size = param.test_size,
                                                    random_state = param.random_state,
                                                    stratify=Y) #  training on 90% Data added stratify on 06/08/21 17:00

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


#########################################
#         Pytorch Model - LSTM          #
#########################################

model = Sequential()
# Doing word embedding
model.add(Embedding(param.MAX_NB_WORDS, param.EMBEDDING_DIM, input_length=X.shape[1]))
# Adding an array Dropout for dropping full vector 
model.add(SpatialDropout1D(param.SpatialDropout1D))
# Adding layer of lstm as hidden layer
model.add(LSTM(param.neurons, dropout=param.dropout, recurrent_dropout=param.recurrent_dropout,return_sequences=True)) #input_shape=X
# Adding another hidden layer to get deep learning
model.add(LSTM(param.neurons,return_sequences=True))
# Adding another hidden layer to get deep learning
model.add(LSTM(param.neurons,return_sequences=True))
# Adding another hidden layer
model.add(LSTM(param.neurons))
# Dense Layer with the number of classes we difined to get
model.add(Dense(param.number_of_classes, activation=param.activation))
# adding a loss function and optimizer and metrics to calculate to get better understanding
model.compile(loss=param.loss_function,
              optimizer=param.optimizer,
              metrics=['accuracy',f1_m,precision_m, recall_m])

print(model.summary())
#  EarlyStopping is for stopping the training if the validation loss going up (Overfitting)!
history = model.fit(X_train,
                    Y_train,
                    epochs=param.epochs,
                    batch_size=param.batch_size,
                    validation_split=param.validation_split,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001),model_checkpoint_callback])#TODO: add function to get the metrics for better understanding

model.save('ltsm_cnn.pth') # saving the model

accr= model.evaluate(X_test,Y_test) # Saves a list of Loss, Accuracy, F1_Score, Precision, Recall

statistics = pd.DataFrame([hist_f1,hist_precision,hist_recall],
                            columns=['F1_Score','Precision','Recall'])

expiriment = pd.DataFrame([[param.epochs,
                            param.batch_size,
                            param.MAX_NB_WORDS,
                            param.MAX_SEQUENCE_LENGTH,
                            param.EMBEDDING_DIM]],
                            columns=['Num_Of_Epochs',
                                     'Batch_Size',
                                     'Max_Num_Words',
                                     'Max_Sequence_Num',
                                     'Embedding_Dim'])

expiriment.to_csv('{}-Expiriment-Data.csv'.format(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")))
statistics.to_csv('{}-Statistics.csv'.format(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")))
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}%\n  F1-Score:{:0.3f}\n  Precision:{:0.3f}\n  Recall:{:0.3f}'.format(accr[0],
                                                                                                                          accr[1]*100,
                                                                                                                          accr[2]*100,
                                                                                                                          accr[3]*100,
                                                                                                                          accr[4]*100))


histo = pd.DataFrame(history.history)
histo.to_csv('History-{}.csv'.format(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")))
#########################################
#                Plot                   #
#########################################


y_pred = model.predict(X_test)
y_true = Y_test
matrix = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
cmd = ConfusionMatrixDisplay(matrix, display_labels=df_label_count_sorted)
cmd.plot()
plt.title('Confusion Matrix')
cmd.ax_.set(xlabel='Predicted', ylabel='Actual')
plt.savefig('{}-Confusion-Matrix'.format(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")))
plt.show()


plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='valid')
plt.legend()
plt.savefig('{}-Loss'.format(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")))
plt.show();



plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='valid')
plt.legend()
plt.savefig('{}-Accuracy'.format(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")))
plt.show();

#########################################
#                Predict                #
#########################################


new_email = ['I am a victim of identity theft and someone stole my identity and personal information to open up a Visa credit card account with Bank of America. The following Bank of America Visa credit card account do not belong to me : XXXX.']
seq = tokenizer.texts_to_sequences(new_email)
padded = pad_sequences(seq, maxlen=param.MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)
print(pred, df_label_count_sorted[np.argmax(pred)])
