import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the saved model
model = load_model('spam_classifier_model.h5')

# Load the saved tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Streamlit UI
st.title('SMS Spam/Ham Classifier')

st.write("""
This is a simple web application that classifies SMS messages as either **Spam** or **Ham**.
Enter a message below to classify it!
""")

# User input
user_input = st.text_area("Enter your message:", "")

# Function to classify input
def classify_message(message):
    # Preprocess the text input just like during model training
    sequences = tokenizer.texts_to_sequences([message])
    padded_sequence = pad_sequences(sequences, maxlen=100)  # Make sure maxlen is the same as used in training

    # Predict the class (0: Ham, 1: Spam)
    prediction = model.predict(padded_sequence)
    predicted_class = np.argmax(prediction, axis=1)

    return predicted_class

if st.button('Classify'):
    if user_input:
        predicted_class = classify_message(user_input)

        # Display the result
        if predicted_class == 1:
            st.subheader("Prediction: Spam")
        else:
            st.subheader("Prediction: Ham")
    else:
        st.warning("Please enter a message to classify.")
