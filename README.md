# News Category Predictor — Streamlit

A small Streamlit interface for the TF-IDF + Dense neural-network model exported
from `News_Categorization_Using_Your_Dataset_FIXED.ipynb`.

## 1. Project structure

After extracting the model ZIP, arrange files like this:

```text
news_category_streamlit/
├── app.py
├── requirements.txt
└── news_category_artifacts/
    ├── news_category_tfidf.keras
    ├── tfidf_vectorizer.joblib
    ├── label_encoder.joblib
    └── model_metadata.json
```

Do not put the ZIP itself inside the artifacts folder. Extract its contents there.

## 2. Run locally

Use Python 3.10 or 3.11 in a fresh virtual environment.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Start the app:
```bash
streamlit run app.py
```

Streamlit will show a local URL, usually http://localhost:8501.

## 3. Important compatibility note

The TF-IDF vectorizer was serialized with joblib in the Colab training
environment. Scikit-learn recommends loading serialized estimators with the
same scikit-learn version used when they were saved. If you see an
`InconsistentVersionWarning` or a vectorizer loading error, check the
scikit-learn version in Colab (`import sklearn; print(sklearn.__version__)`)
and install that same version locally/deployment.

For a reproducible deployment, record the Colab versions of TensorFlow,
scikit-learn, NumPy and joblib after training, then pin them in requirements.

## 4. Deploy

You can deploy this Streamlit app on Streamlit Community Cloud by pushing this
folder (including the extracted model artifact folder) to a GitHub repository,
then selecting `app.py` as the app entry point.

Netlify hosts static/frontend applications; it does not directly host this
Streamlit Python app. If you want to keep Netlify, use a separate Python API
(e.g. FastAPI) for inference and let the Netlify frontend call that API.

## 5. Model behavior

The app uses the same lowercasing, URL/email/hashtag/punctuation/number
removal, English stopword removal and WordNet lemmatization described in the
training notebook. It shows the top predicted categories and their probability
scores. The model always chooses among its trained categories, even for
headlines outside those topics.
