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
#Explainability
def explain_prediction(text, model, vectorizer, top_n=6):
    X = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    non_zero_indices = X.nonzero()[1]
    contributions = []

    for idx in non_zero_indices:
        word = feature_names[idx]
        tfidf_value = X[0, idx]
        weight = coefficients[idx]
        contribution = tfidf_value * weight
        contributions.append((word, contribution))

    contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
    return contributions[:top_n]

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
        #EXplainability
        st.markdown("### 🔍 Model Explanation")
        st.caption("Words that influenced the prediction the most")

        explanation = explain_prediction(text_input, model, vectorizer)

        if explanation:
            for word, score in explanation:
                if score > 0:
                    st.markdown(f"🟥 **{word}** → {score:.4f}")
                else:
                    st.markdown(f"🟩 **{word}** → {score:.4f}")
        else:
            st.write("No significant words found.")

# Footer
st.markdown("---")
st.caption("Model: TF-IDF + Logistic Regression | Explainable ML")

