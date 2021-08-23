# -- coding: utf-8 --
"""
Created on Sun Aug 15 23:29:06 2021

@author: Amir Hoshen

GUI Aplication for model predictions
"""
try:
    from tkinter import *
    import tkinter as tk
    import pandas as pd
    from tkinter import font 
    from keras.preprocessing.text import Tokenizer
    import lstm_parameters as param
    from lstm_parameters import MAX_SEQUENCE_LENGTH,MAX_NB_WORDS,loss_function,optimizer
    from tensorflow.keras.models import model_from_json
    from preprocessEmailMessage import *
    from sklearn.model_selection import train_test_split
    import numpy as np
    np.random.seed(42)
    from keras.preprocessing.sequence import pad_sequences
    import pickle

except Exception as e:
    print("Some Modules are missing {}".format(e))


#GUI window 
window = Tk()

#GUI elements

#GUI window title
window.title("Phishing Email Detection")

#GUI window size
window.geometry("800x500+150+150")

#window background color
window.configure(background="DeepSkyBlue4")

#Font object
myFont = font.Font(family='Helvetica', weight='bold')

#Email message instruction label
label1 = Label(window, text='Сopy&Past E-mail Message',bg='DeepSkyBlue4', fg='gold', font=(myFont, 18))
label1.pack(padx=0, pady=0, side=tk.TOP)

hist_recall = []
hist_precision = []
hist_f1 = []

#Dataframe object 
df = pd.read_csv('final_dataset.csv')
df = df.drop(columns=['Unnamed: 0'])
df_label_count = df.Label.value_counts()
# creates label list by alphabetical order
df_label_count_sorted = sorted(df_label_count.index.values, key=lambda s: s.lower())  

#Model loading
json_file = open('architecture.json', 'r')
model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("my_model_weights.h5")
tokenizer,data,num_words,maxlen,Y = None,None,None,None,None
with open("tokenizer.pkl", 'rb') as f:
    data = pickle.load(f)
    tokenizer = data['tokenizer']
    num_words = data['num_words']
    maxlen = data['maxlen']
    Y = data['Y']
X = tokenizer.texts_to_sequences(df['Email'].values)
X = pad_sequences(X, maxlen=maxlen)

#train test split
X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size = 0.1,
                                                    random_state = 42,
                                                    stratify=Y)
#Compiling the model
model.compile(loss=param.loss_function,
              optimizer=optimizer,
              metrics=['accuracy',param.f1_m, param.precision_m, param.recall_m])
#Model evaluation
accr= model.evaluate(X_test,Y_test)


#GUI Entry box
textbox1 = Entry(window, width=50, bg='white', fg='royal blue', font=(myFont, 25))
textbox1.place(x=10, y=10)
textbox1.pack(padx=10, pady=10)

def clear_view_button():
    '''
    Clear view button from GUI, once pressed all values will be cleaned from screen
    including GUI entry box.
    Returns
    -------
    None.

    '''
    pre_ans_label.destroy()
    pred_visual_label.destroy()
    textbox1.delete(0,'end')
    button_submit['state'] = NORMAL
    
    
def submit_button():
    '''
    GUI Submition button, once pressed a prediction will be made over the message past into the entry box,
    then prediction table will be shown on GUI a long with the maximum precentage lable(spam, normal, phish, fraud) 
    including the model accuracy.
    Returns
    -------
    None.

    '''
    global pre_ans_label, pred_visual_label, textbox1
    _fg ='green2'
    txt = "no text"
    email_content = textbox1.get()
    #TO-DO integrate text onto the predictive model  and return answers and statistic information.
    
    #if messsage input is empty labled it onto th GUI.
    if len(email_content)==0:
        txt=''
        pred_txt = "No E-mail inserted!!! Please press on Clear view button & try again.."
        _fg = 'red'
        _fg_secnd = 'red'
    else:
        _fg_secnd = 'gold'
        word_to_predict = []
        email_content = clean_text(email_content)
        word_to_predict.append(email_content)
        seq = tokenizer.texts_to_sequences(word_to_predict)
        padded = pad_sequences(seq, maxlen=maxlen)
        pred = model.predict(padded)
        txt = '\n'+ str(df_label_count_sorted[np.argmax(pred)]) + str('\n\n{:0.3f}%'.format(pred[0][np.argmax(pred)]*100))
        
        pred_txt = str('\nPrediction Table:\n\nFraud:{:0.3f}%\tNoraml:{:0.3f}%\tPhish:{:0.3f}%\tSpam:{:0.3f}%'.format(pred[0][0]*100,
                                                                pred[0][1]*100,
                                                                pred[0][2]*100,
                                                                pred[0][3]*100))
        
        ans =  str(df_label_count_sorted[np.argmax(pred)])
        print(txt, '{:0.3f}%'.format(float(pred[0][np.argmax(pred)])*100))
        print(email_content)
       
        if ans == 'NORMAL':
            _fg ='green2'
        elif ans == 'FRAUD':
            _fg ='orange'
        elif ans == 'PHISH':
            _fg ='red'
        else:
            _fg ='white'
    
    pre_ans_label = Label(window, text=txt, bg='DeepSkyBlue4', fg= _fg, font=(myFont, 18))
    pred_visual_label = Label(window, text=pred_txt, bg='DeepSkyBlue4', fg=_fg_secnd, font=(myFont, 18))
    
    
    #ans_label = Label(window, text=ans , bg='gold', fg='black', font=(myFont, 14))
    pre_ans_label.pack(pady=10)
    pred_visual_label.pack(pady=20)
    #ans_label.pack(pady=30)
    button_submit['state'] = DISABLED

#Submition button
button_submit = Button(window, text='Submit',command=submit_button, bg='seashell4', fg='gold', font=(myFont, 20))
button_submit.place(x=10,y=30)
button_submit.pack()

#Clear view button
button_clear = Button(window, text="Clear view", command=clear_view_button, bg='seashell4', fg='gold', font=(myFont, 14))
button_clear.place(x=20, y=20)
button_clear.pack(side=tk.BOTTOM)


window.mainloop()