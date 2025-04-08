# import libraies and models

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb # type: ignore
from tensorflow.keras.preprocessing import sequence # type: ignore
from tensorflow.keras.models import load_model # type: ignore


## Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index= {value:key for key ,value in  word_index.items()}


# Load the Pre-trained model with   ReLU activation
model = load_model('simple_rnn_imdb.h5')


## helper Functions
## Function to decode reviews
def decode_review(encode_review):
    return ' '.join([reverse_word_index.get(i-3,'?')for i in encode_review])
# function to preprocess use input
def preprocess_text(text):
    words = text.lower().split()
    encode_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encode_review],maxlen=500)
    return padded_review


## Step : Prediction Function
def predict_sentiment(review):
    preprocesed_input = preprocess_text(review)

    prediction =model.predict(preprocesed_input)

    sentiment = 'Positive' if prediction[0][0]>0.5 else 'Negative'

    return sentiment,prediction[0][0]


import streamlit as st
## Streamlit app
st.title('IMDB Moview Review sentiment Analysis')
st.write('Enter a movie review to classify it is a positive or neagative')

# user input
user_input  = st.text_area('movie_review')

if st.button('classify'):
    preprocessed_input  =preprocess_text(user_input)

    # make my predition
    prediction = model.predict(preprocessed_input)
    sentiments = 'Positive' if prediction[0][0] > 0.5 else 'Negative'


    # Display the Result

    st.write(f'Sentiment:{sentiments}')
    st.write(f'prediction Score:{prediction[0][0]}')
else:
    st.write('please enter a moview review')
