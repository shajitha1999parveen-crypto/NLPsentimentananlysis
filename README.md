# NLP Toolkit — Interactive Streamlit Demo

An interactive Streamlit app demonstrating a full classic-to-modern NLP
pipeline: tokenization, stopword removal, stemming, lemmatization,
Bag-of-Words / TF-IDF vectorization, POS tagging, Named Entity Recognition,
a Naive Bayes text classifier, Word2Vec embeddings, a PyTorch LSTM
classifier architecture, and pretrained Hugging Face Transformer pipelines
(sentiment analysis + text generation).

## Project structure

```
nlp-toolkit/
├── app.py              # Streamlit UI (7 tabs, one per NLP technique)
├── nlp_utils.py         # All NLP logic, kept separate from the UI
├── requirements.txt      # Python dependencies
├── .streamlit/
│   └── config.toml      # App theme/server config
├── .gitignore
└── README.md
```

## 1. Run it locally in VS Code

```bash
# 1. Open the folder in VS Code
code nlp-toolkit

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

> First run will also download small NLTK corpora (punkt, stopwords, POS
> tagger) and, the first time you use the Transformers tab, pretrained
> model weights (`distilbert` for sentiment, `gpt2` for generation).

## 2. Push to GitHub

From inside the `nlp-toolkit` folder:

```bash
git init
git add .
git commit -m "Initial commit: NLP toolkit Streamlit app"

# Create an empty repo on GitHub first (via github.com or `gh repo create`),
# then point your local repo at it:
git branch -M main
git remote add origin https://github.com/<your-username>/nlp-toolkit.git
git push -u origin main
```

If you use the GitHub CLI instead:

```bash
gh repo create nlp-toolkit --public --source=. --remote=origin --push
```

## 3. Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Click **"New app"**.
3. Pick your `nlp-toolkit` repo, branch `main`, and main file path `app.py`.
4. Click **Deploy**.

Notes:
- Streamlit Cloud installs everything in `requirements.txt` automatically,
  including the spaCy model wheel URL, so `en_core_web_sm` will be
  available without a separate download step.
- The free tier has limited RAM/CPU. `torch` + `transformers` + `gpt2`
  are heavy — first load of the Transformers tab may be slow, and on the
  smallest free instances it can hit memory limits. If that happens,
  either upgrade the resource tier or drop the GPT-2 generation feature
  and keep only the lighter sentiment pipeline.
- Because model downloads happen at runtime, first load after each app
  restart will be slower ("cold start").

## Algorithms & techniques used

| Area | Technique / Algorithm |
|---|---|
| Tokenization | NLTK `word_tokenize`, `sent_tokenize` (Punkt tokenizer) |
| Text cleaning | Stopword removal (NLTK stopword list) |
| Normalization | Porter Stemmer (NLTK); Lemmatization (spaCy statistical pipeline) |
| Feature extraction | Bag-of-Words (`CountVectorizer`); TF-IDF (`TfidfVectorizer`) |
| Sequence tagging | POS tagging (NLTK averaged perceptron tagger) |
| Information extraction | Named Entity Recognition (spaCy `en_core_web_sm`) |
| Classification | Multinomial Naive Bayes (`sklearn.naive_bayes`) |
| Evaluation | Accuracy, Precision, Recall, F1-score (`sklearn.metrics`) |
| Word embeddings | Word2Vec (`gensim`, skip-gram/CBOW) |
| Deep learning | LSTM (Embedding → LSTM → Linear) sequence classifier in PyTorch |
| Transfer learning | Pretrained Transformer pipelines — DistilBERT for sentiment analysis, GPT-2 for text generation (Hugging Face `transformers`) |

## Suggested resume bullet points

> **NLP Toolkit — Interactive NLP Pipeline (Python, Streamlit, PyTorch, Hugging Face)**
- Built and deployed an end-to-end NLP demo application covering
  preprocessing (tokenization, stopword removal, stemming, lemmatization),
  feature engineering (Bag-of-Words, TF-IDF), and sequence labeling
  (POS tagging, Named Entity Recognition) using NLTK and spaCy.
- Implemented a Multinomial Naive Bayes text classifier with a full
  train/test evaluation workflow (accuracy, precision, recall, F1) using
  scikit-learn.
- Designed a Word2Vec embedding model (gensim) and a PyTorch LSTM
  (Embedding–LSTM–Linear) architecture for sequence classification.
- Integrated pretrained Hugging Face Transformer pipelines (DistilBERT
  sentiment analysis, GPT-2 text generation) for transfer-learning-based
  inference.
- Packaged the project as an interactive multi-tab Streamlit web app,
  version-controlled with Git/GitHub, and deployed publicly on Streamlit
  Community Cloud.

Feel free to trim this to 2–3 bullets depending on the resume section
(Projects vs. Experience) and how much space you have.
