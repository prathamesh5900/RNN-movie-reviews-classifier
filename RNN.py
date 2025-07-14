import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Constants
MAX_WORDS = 10000  # Must match the model's training configuration
MAX_LEN = 500      # Input length for padding

# Load the IMDb word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the trained model
model = load_model('simple_rnn_imdb.h5')

# Preprocess input text with word filtering
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [
        word_index.get(word, 2) + 3  # offset for special tokens
        for word in words
        if word_index.get(word, 2) + 3 < MAX_WORDS  # filter out unknowns
    ]
    padded_review = sequence.pad_sequences(
        [encoded_review], maxlen=MAX_LEN, dtype='int32'
    )
    return padded_review

# Predict sentiment
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Streamlit UI
st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.write("Enter a movie review and classify it as **Positive** or **Negative**")

# Input box
user_input = st.text_area("✏️ Your Review")

if st.button("🚀 Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a review before clicking Classify.")
    else:
        sentiment, score = predict_sentiment(user_input)
        st.subheader("🧠 Prediction Result")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Prediction Score:** {score:.4f}")
