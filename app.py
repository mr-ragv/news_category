import json
import re
from pathlib import Path

import joblib
import nltk
import numpy as np
import streamlit as st
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

APP_DIR = Path(__file__).parent
ARTIFACT_DIR = APP_DIR / "news_category_artifacts"

st.set_page_config(
    page_title="News Category Predictor",
    page_icon="📰",
    layout="centered",
)

@st.cache_resource
def load_nltk_resources():
    # Download only if the resources are not already present.
    for package, resource in [
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)

    return set(stopwords.words("english")), WordNetLemmatizer()

@st.cache_resource
def load_artifacts():
    model_path = ARTIFACT_DIR / "news_category_tfidf.keras"
    vectorizer_path = ARTIFACT_DIR / "tfidf_vectorizer.joblib"
    encoder_path = ARTIFACT_DIR / "label_encoder.joblib"
    metadata_path = ARTIFACT_DIR / "model_metadata.json"

    missing = [
        p.name for p in [model_path, vectorizer_path, encoder_path, metadata_path]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(missing)
            + ". Extract news_category_artifacts.zip into the project folder."
        )

    model = tf.keras.models.load_model(model_path, compile=False)
    vectorizer = joblib.load(vectorizer_path)
    label_encoder = joblib.load(encoder_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, vectorizer, label_encoder, metadata

def clean_text(text, stop_words, lemmatizer):
    # Keep this preprocessing identical to the notebook's training pipeline.
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\S*@\S*\s?", "", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

st.title("📰 News Category Predictor")
st.write(
    "Enter a news headline and the trained machine-learning model "
    "will predict its most likely category."
)

try:
    with st.spinner("Loading trained model..."):
        model, vectorizer, label_encoder, metadata = load_artifacts()
        stop_words, lemmatizer = load_nltk_resources()
except Exception as exc:
    st.error("The model could not be loaded.")
    st.code(str(exc))
    st.info(
        "Expected structure: this app.py file beside a folder named "
        "'news_category_artifacts' containing the four exported model files."
    )
    st.stop()

with st.expander("About this model"):
    st.write("**Model:**", metadata.get("model_name", "TF-IDF + Dense Neural Network"))
    st.write("**Number of categories:**", len(label_encoder.classes_))
    if "test_accuracy" in metadata:
        st.metric("Test accuracy", f"{metadata['test_accuracy'] * 100:.2f}%")
    if "test_macro_f1" in metadata:
        st.metric("Test macro F1", f"{metadata['test_macro_f1']:.4f}")

headline = st.text_area(
    "News headline",
    placeholder="Example: Central bank announces a new policy to control inflation",
    height=110,
)

top_k = st.slider(
    "Number of predictions to show",
    min_value=1,
    max_value=min(5, len(label_encoder.classes_)),
    value=min(3, len(label_encoder.classes_)),
)

if st.button("Predict category", type="primary", use_container_width=True):
    if not headline.strip():
        st.warning("Please enter a headline first.")
    else:
        cleaned = clean_text(headline, stop_words, lemmatizer)
        if not cleaned:
            st.warning("The headline contains no usable words after preprocessing.")
        else:
            features = vectorizer.transform([cleaned]).toarray().astype(np.float32)
            probabilities = model.predict(features, verbose=0)[0]
            indices = np.argsort(probabilities)[::-1][:top_k]
            predicted_labels = label_encoder.inverse_transform(indices)

            st.subheader("Prediction")
            st.success(f"Most likely category: **{predicted_labels[0]}**")
            st.caption(
                "Confidence scores are the model's predicted probabilities, "
                "not a guarantee that the category is correct."
            )

            for label, idx in zip(predicted_labels, indices):
                score = float(probabilities[idx])
                st.write(f"**{label}** — {score * 100:.2f}%")
                st.progress(score)

            st.caption(
                "This model was trained on the selected news categories from "
                "your uploaded dataset. Predictions outside those categories "
                "will still be assigned to one of the available categories."
            )

st.divider()
st.caption("News Category Classification • NLP project showcase")
