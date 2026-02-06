import streamlit as st
import pickle
import numpy as np

# Load model and vectorizer
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("logistic_hate_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_artifacts()

# Page config
st.set_page_config(
    page_title="Hate Speech Detector",
    page_icon="⚠️",
    layout="centered"
)

# UI
st.title("🚨 Hate Speech Detection")
st.write("Enter any text below to check whether it contains hateful speech.")

text_input = st.text_area(
    "Text Input",
    height=150,
    placeholder="Type or paste text here..."
)

# Prediction
if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        X = vectorizer.transform([text_input])
        prediction = model.predict(X)[0]
        confidence = model.predict_proba(X).max() * 100

        if prediction == 1:
            st.error(f"⚠️ Hateful Speech Detected\n\nConfidence: {confidence:.2f}%")
        else:
            st.success(f"✅ Not Hateful\n\nConfidence: {confidence:.2f}%")

# Footer
st.markdown("---")
