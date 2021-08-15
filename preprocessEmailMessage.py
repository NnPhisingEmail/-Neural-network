# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: Amir Hoshen 

this file is set to preprocess emails message content in order to
transform the data onto GUI for a predictive machine learning model.

"""


import string
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
nltk.download('punkt')

email_pred_test = "I am a victim of identity theft and someone stole my identity and personal information to open up a Visa credit card account with Bank of America. The following Bank of America Visa credit card account do not belong to me : XXXX."
email_test = 'the, is, are \t\d\d\d\d\d\d\d \d \\ \<<link>><<link>><<link>>\ndear\n \nnational city\n business client ,national city ,corporate customer service requests complete national city business online client form procedure obligatory business corporate clients national city please select hyperlink domain com address listed access national city business online client form thank choosing national city business needs look forward working please respond email mail generated automated service replies mail read national city corporate customer service technical support x x x x x x x x x xhnq x x x x x x oiix x x x x x x x x x x x x x x x x x oe x x x x x x x x x x x x x x xt x x x x x x x x wlt x x x x x x x x x x x x x x x x x x x cvs sck close kh cvs x x x x x x x x x jem serv sbdx rcs tb od f x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x xx x x x x x x x x x x x ua x x x x qo x x x x x x x x x x source hk xh source exe cvs nxq x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x file opjq common nl function define exe x x x x x'


#nltk.download('punkt')
def remove_punct(text):
  '''
    

    Parameters
    ----------
    text : string
        Taking the Email body message and removing punctuation from it.

    Returns
    -------
    text_nopunct : string
        no punctuation string.

    '''
  text_nopunct =''
  text_nopunct = re.sub('['+string.punctuation+']', ' ' ,text)
  text_nopunct.strip()
  return text_nopunct



#nltk.download('stopwords')
def removeStopWords(tokens):
    '''
    

    Parameters
    ----------
    tokens : string
        text to iterate over while removing all stopwords.

    Returns
    -------
    string
        list of fixed tex with no stopWords.

    '''
    text_tokens = word_tokenize(tokens)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    text_no_sw = ''
    for word in tokens_without_sw:
        text_no_sw += word+' '
    
    #print(text_no_sw)
    return text_no_sw



def clean_text(text):
    '''
    Parameters
    ----------
    text : is a simple String object.
        Method gets a string of an Email body message preprocess the message and get it ready for machine learning
        model, main usage for this function is for GUI message upload.

    Returns
    -------
    String ready for model prediction .

    '''
    print('clean_text started')
    if(text != "" and text != " "):
        print('Inside if')
        p = re.compile('(\n| x|[\t]*|<<link>>|a0|\\\\d+|\\\\)')
        
        fixed_text = text
        fixed_text.lower()
        fixed_text = p.sub('', fixed_text, count=0)
        fixed_text = remove_punct(fixed_text)
        fixed_text = removeStopWords(fixed_text)
        return fixed_text
    else:
        print("The input text for function 'clean_text' is empty!, check inputes and try again...") 
        return text



'''
run example from this script bellow:

print('Input Email Text:\n',email_test)
email_test = clean_text(email_test)
print('\nOutput Email Text:\n',email_test)
'''
