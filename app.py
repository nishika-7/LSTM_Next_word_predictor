"""
Streamlit app for the LSTM Next-Word Predictor (trained on Shakespeare's Hamlet).
"""

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = "next_word_lstm.keras"
TOKENIZER_PATH = "tokenizer.pickle"

# Training used pad_sequences(input_sequences, maxlen=14, ...) to build the
# full sequences, then split x = input_sequences[:, :-1] (length 13) and
# y = input_sequences[:, -1]. The Embedding layer used
# input_length = max_sequence_len - 1, so the model actually expects 13
# tokens as input, not 14. If predictions raise a shape-mismatch error,
# try changing this back to 14.
MAX_SEQUENCE_LEN = 13

st.set_page_config(page_title="LSTM Next Word Predictor", page_icon="🔮")


@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


model, tokenizer = load_artifacts()


def predict_next_words(seed_text, num_words):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=MAX_SEQUENCE_LEN, padding="pre")
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = int(np.argmax(predicted_probs, axis=-1)[0])

        next_word = next(
            (word for word, idx in tokenizer.word_index.items() if idx == predicted_index),
            ""
        )
        if not next_word:
            break
        seed_text += " " + next_word
    return seed_text


st.title("🔮 LSTM Next Word Predictor")
st.caption("Trained on Shakespeare's Hamlet")
st.write("Enter a phrase and the model will predict what comes next.")

seed_text = st.text_input("Starting text:", "I love")
num_words = st.slider("Words to predict:", 1, 10, 1)

if st.button("Predict"):
    with st.spinner("Thinking..."):
        result = predict_next_words(seed_text, num_words)
    st.success(result)
