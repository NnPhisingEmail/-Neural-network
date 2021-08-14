# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 20:05:31 2021

@author: Peleg
"""
#########################################
#               Imports                 #
#########################################
import lstm_parameters as param
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping,ModelCheckpoint
#from nltk.corpus import stopwords
#from nltk import word_tokenize
#STOPWORDS = set(stopwords.words('english'))
import cufflinks
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras import backend as K
import datetime
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
