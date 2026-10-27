import json
import pickle
from pathlib import Path

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

ARTIFACTS = Path("artifacts")

st.set_page_config(page_title="Hamlet Next-Word Predictor", page_icon="📖")
st.title("Next-Word Prediction")
st.caption("A learning project using an LSTM trained on Shakespeare's Hamlet.")

required_files = [
    ARTIFACTS / "next_word_lstm.keras",
    ARTIFACTS / "tokenizer.pkl",
    ARTIFACTS / "config.json",
]
if not all(path.exists() for path in required_files):
    st.error("Model files are missing. Run `python train_model.py` first, then commit the artifacts folder.")
    st.stop()

@st.cache_resource
def load_artifacts():
    model = load_model(ARTIFACTS / "next_word_lstm.keras")
    with open(ARTIFACTS / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open(ARTIFACTS / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    return model, tokenizer, config

model, tokenizer, config = load_artifacts()
max_context = int(config["max_context"])
vocab_size = int(config["vocab_size"])

def predict_top_words(text, top_k=5):
    sequence = tokenizer.texts_to_sequences([text.lower()])[0]
    # Ignore empty input and use only the most recent context words.
    sequence = sequence[-max_context:]
    if not sequence:
        return []

    padded = pad_sequences([sequence], maxlen=max_context, padding="pre", truncating="pre")
    probabilities = model.predict(padded, verbose=0)[0]

    # Exclude padding index 0 from predictions.
    probabilities[0] = 0
    best_indices = probabilities.argsort()[-top_k:][::-1]

    index_to_word = {index: word for word, index in tokenizer.word_index.items()}
    return [
        (index_to_word.get(int(index), "<unknown>"), float(probabilities[index]))
        for index in best_indices
        if probabilities[index] > 0
    ]

input_text = st.text_input("Enter a phrase", value="To be or not to be")
top_k = st.slider("Number of suggestions", min_value=1, max_value=10, value=5)

if st.button("Predict next word", type="primary"):
    if not input_text.strip():
        st.warning("Enter at least one word.")
    else:
        predictions = predict_top_words(input_text, top_k=top_k)
        if not predictions:
            st.info("No prediction could be generated for this input.")
        else:
            st.subheader("Suggested next words")
            for word, probability in predictions:
                st.write(f"**{word}** — {probability:.1%}")
            st.caption("Probabilities are model scores, not guarantees of correctness.")