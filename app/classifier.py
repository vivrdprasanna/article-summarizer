#imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle

# helps in text preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

#load model and tokenizer
model = keras.models.load_model('spam_model')
with open('x_tokenizer.pickle', 'rb') as handle:
    x_tokenizer = pickle.load(handle)

def predict_with_percent(sms_original):
    #convert to vectors
    sms = np.array([sms_original])
    sms_seq = x_tokenizer.texts_to_sequences(sms)
    sms_padded_seq = pad_sequences(sms_seq, maxlen = 8, padding = 'post')

    #prediction
    pred = (model.predict(sms_padded_seq) > 0.5).astype("int32").item()
    certainty = 100 * float(model.predict(sms_padded_seq))

    #output
    label = 'spam' if pred == 1 else 'not spam!'
    percent = certainty
    return label, percent