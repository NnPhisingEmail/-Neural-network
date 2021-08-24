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
    import tkinter.font as font
    from tkinter import filedialog as fd
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

'''
if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
'''
  
window = Tk()
#GUI elements

#GUI window size
window.title("Phishing Email Detection")

window.geometry("800x500+150+150")

#window background color
window.configure(background="DeepSkyBlue4")

#create Font object
myFont = font.Font(family='Helvetica', weight='bold')

#Email message instruction label
label1 = Label(window, text='Сopy & past e-mail message',bg='DeepSkyBlue4', fg='gold', font=(myFont, 18))
label1.pack(padx=0, pady=0, side=tk.TOP)

data = StringVar()

hist_recall = []
hist_precision = []
hist_f1 = []


df = pd.read_csv('final_dataset.csv')
df = df.drop(columns=['Unnamed: 0'])
df_label_count = df.Label.value_counts()
df_label_count_sorted = sorted(df_label_count.index.values, key=lambda s: s.lower())  # creates label list by alphabetical order


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

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size = 0.1,
                                                    random_state = 42,
                                                    stratify=Y)
model.compile(loss=param.loss_function,
              optimizer=optimizer,
              metrics=['accuracy',param.f1_m, param.precision_m, param.recall_m])

accr= model.evaluate(X_test,Y_test)


filePath = ''
textbox1 = Entry(window, width=50, bg='white', fg='royal blue', font=(myFont, 25))
textbox1.place(x=10, y=10)
textbox1.pack(padx=10, pady=10)
bColor = 'gold'
def myDelete():
    global filePath
    filePath = ''
    pre_ans_label.destroy()
    file_button.configure(fg='gold')
    
    button_submit['state'] = NORMAL
    
def get_file_path():
    global bColor
    global filePath
    bColor = 'gold'
    filetypes = (
        ('text files', '*.txt'),
        ('eml files', '*.eml')
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)

    if len(filename) == 0:
        bColor = 'gold'
    else:
        bColor = 'green2'
        filePath = filename
    file_button.configure(fg=bColor)
        
    return filename

def button_command():
    global pre_ans_label
    global filePath
    
    _fg ='green2'
    txt = "no text"
    email_content = textbox1.get()
    #TO-DO integrate text onto the predictive model  and return answers and statistic information.
    
    #if messsage input is empty labled it onto th GUI.
    if len(email_content)==0 and len(filePath)==0:
        txt = "No e-mail inserted!! try again.."
    elif len(filePath) != 0 and len(email_content) == 0:
        word_to_predict = []
        with open(filePath,'r',encoding='utf8') as f:
            content = get_content(f)
            print(content)
            clean_content = clean_text(content)
            print(clean_content)
            word_to_predict.append(clean_content)
            seq = tokenizer.texts_to_sequences(word_to_predict)
            padded = pad_sequences(seq, maxlen=maxlen)
            pred = model.predict(padded)
            txt = str(df_label_count_sorted[np.argmax(pred)]) + str('\n{:0.3f}%\n\nPrediction Table:\nFraud:{:0.3f}%\tNoraml:{:0.3f}%\tPhish:{:0.3f}%\tSpam:{:0.3f}%'.format(pred[0][np.argmax(pred)]*100,
                                                                pred[0][0]*100,
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
    elif len(email_content) != 0 and len(filePath)==0 :
        word_to_predict = []
        email_content = clean_text(email_content)
        word_to_predict.append(email_content)
        seq = tokenizer.texts_to_sequences(word_to_predict)
        padded = pad_sequences(seq, maxlen=maxlen)
        pred = model.predict(padded)
        txt = str(df_label_count_sorted[np.argmax(pred)]) + str('\n{:0.3f}%\n\nPrediction Table:\nFraud:{:0.3f}%\tNoraml:{:0.3f}%\tPhish:{:0.3f}%\tSpam:{:0.3f}%'.format(pred[0][np.argmax(pred)]*100,
                                                                pred[0][0]*100,
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
    else:
        txt = 'Two Options In Use Please Press On "Clear View" and Try Again'
    pre_ans_label = Label(window, text=txt, bg='DeepSkyBlue4', fg= _fg, font=(myFont, 18))
    
    textbox1.delete(0,'end')
    #ans_label = Label(window, text=ans , bg='gold', fg='black', font=(myFont, 14))
    pre_ans_label.pack(pady=10)
    #ans_label.pack(pady=30)
    button_submit['state'] = DISABLED

file_button = Button(window, text='Choose File',command=get_file_path, bg='seashell4', fg=bColor, font=(myFont, 20))
file_button.place(x=30,y=30)
file_button.pack()

button_submit = Button(window, text='Submit',command=button_command, bg='seashell4', fg='gold', font=(myFont, 20))
button_submit.place(x=10,y=30)
button_submit.pack()

button_clear = Button(window, text="Clear view", command=myDelete, bg='seashell4', fg='gold', font=(myFont, 14))
button_clear.place(x=20, y=20)
button_clear.pack(side=tk.BOTTOM)


window.mainloop()