from attention import AttentionLayer
import re
import sys
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")
from tensorflow.keras.models import load_model

from utils_for_input_to_summary import text_cleaner

# # Loading the model
# def get_model():
#     from tensorflow.keras.models import load_model
#     summarizer = load_model('ACTUAL_MODEL_PLEASE_WORK', compile=True)
#     print("==============================================")
#     print("model loaded successfully.", "\n")
#     return summarizer

# text = 'Gave caffeine shakes heart anxiety attack plus tastes unbelievably bad stick coffee tea soda thanks!!'


# text = [str(input('text: '))]

text = [str('Gave caffeine shakes heart anxiety attack plus tastes unbelievably bad stick coffee tea soda thanks!!')]

# clean the text
cleaned_text = []
for t in text:
    cleaned_text.append(text_cleaner(t,0)) 
    
print(len(cleaned_text))
print(cleaned_text)


# tokenize the text into vectors
cnt=4
tot_cnt=9089
max_text_len=30


tokenizer = Tokenizer(num_words=tot_cnt-cnt)
tokenizer.fit_on_texts(cleaned_text)
cleaned_text = tokenizer.texts_to_sequences(cleaned_text)
tt = tf.keras.preprocessing.sequence.pad_sequences(cleaned_text, padding='post', maxlen=max_text_len)
print(tt)


# Loading the model
def get_model():
    from tensorflow.keras.models import load_model
    summarizer = load_model('ACTUAL_MODEL_PLEASE_WORK', compile=True)
    print("==============================================")
    print("model loaded successfully.", "\n")
    return summarizer

from utils_for_input_to_summary import get_decode_function
decode = get_decode_function(36, 100)
print("successfully imported decoding function.")
max_length = 35
summary = decode(sequence.reshape(1,max_length))
print("summary:", summary)



