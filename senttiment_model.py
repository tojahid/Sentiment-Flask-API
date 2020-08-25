from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# import  tensorflow_datasets as tfds
from tensorflow import keras as k
import numpy as np 
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import nltk
import re
import pandas as pd


model = k.models.load_model('./weights/sentiemnt_tf.h5')

def Test_Text_Process(text):
    
    training_sentences = list()
    training_sentences.append(text)
    # tokenize sentences
    vocab_size = 10000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    oov_tok = "<OOV>"



    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)
    
    return padded 

def classify(text):
    text = Test_Text_Process(text)
    LABELS = ['Neutral', 'Negative', 'Positive']

    y_pred = model.predict(text)
    label_index = np.argmax(y_pred)
    output = LABELS[label_index]

    return output


if __name__ == '__main__':
    text = "ভোদাই মমিনুল রে নিয়া হাথুরুকে প্রশ্ন করছে সে একটা খাটি চোদনা"
    output = classify(text)
    print(output)