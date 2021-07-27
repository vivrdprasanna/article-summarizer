import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import wordcloud
import pickle

# helps in text preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# helps in model building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import EarlyStopping


model = tf.keras.models.load_model("spam_model.h5")
with open('tokenizer.pkl', 'rb') as input:
    tokenizer = pickle.load(input)
    
def get_spam_or_ham(pred):
    if pred == 1:
        return 'spam!'
    return 'not spam!'

def predict(sms_original):
    sms = np.array([sms_original])
    sms_proc = tokenizer.texts_to_sequences(sms)
    sms_proc = pad_sequences(sms_proc, maxlen = 8, padding='post')
    pred = (model.predict(sms_proc) > 0.5).astype("int32").item()
    return get_spam_or_ham(pred)


    
    

    