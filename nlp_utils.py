"""
nlp_utils.py
------------
Reusable NLP building blocks used by the Streamlit app (app.py).

Covers:
- Tokenization (word + sentence)
- Stopword removal
- Stemming (Porter) and Lemmatization (spaCy)
- Bag-of-Words and TF-IDF vectorization
- POS tagging
- Named Entity Recognition (spaCy)
- Naive Bayes text classification + evaluation metrics
- Word2Vec embeddings (gensim)
- A simple LSTM classifier (PyTorch)
- Pretrained transformer pipelines (sentiment analysis, text generation)
"""

import nltk
import spacy
import torch
import torch.nn as nn

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from gensim.models import Word2Vec
from transformers import pipeline


# ---------------------------------------------------------------------------
# One-time NLTK / spaCy resource setup
# ---------------------------------------------------------------------------
def ensure_nltk_resources():
    """Download required NLTK corpora if they aren't already present."""
    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    ]
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            # Some resource names vary across NLTK versions; safe to ignore.
            pass


_NLP = None


def get_spacy_model():
    """Load (and cache) the spaCy English model."""
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def tokenize(text: str):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    return words, sentences


def remove_stopwords(words):
    stop = set(stopwords.words("english"))
    return [w for w in words if w.lower() not in stop]


def stem_words(words):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in words]


def lemmatize_text(text: str):
    nlp = get_spacy_model()
    doc = nlp(text)
    return [token.lemma_ for token in doc]


# ---------------------------------------------------------------------------
# Vectorization
# ---------------------------------------------------------------------------
def bag_of_words(corpus):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    return vectorizer.get_feature_names_out(), X.toarray()


def tfidf_vectorize(corpus):
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(corpus)
    return tfidf.get_feature_names_out(), X.toarray()


# ---------------------------------------------------------------------------
# POS tagging & NER
# ---------------------------------------------------------------------------
def pos_tag_text(text: str):
    tokens = word_tokenize(text)
    return nltk.pos_tag(tokens)


def named_entities(text: str):
    nlp = get_spacy_model()
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


# ---------------------------------------------------------------------------
# Naive Bayes text classification
# ---------------------------------------------------------------------------
def train_naive_bayes(texts, labels, test_size=0.25, random_state=42):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=random_state
    )
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, vectorizer, y_test, y_pred


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Word2Vec embeddings
# ---------------------------------------------------------------------------
def train_word2vec(tokenized_sentences, vector_size=50, window=2, min_count=1):
    model = Word2Vec(
        tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
    )
    return model


# ---------------------------------------------------------------------------
# LSTM classifier (PyTorch)
# ---------------------------------------------------------------------------
class LSTMModel(nn.Module):
    """A minimal LSTM text classifier: Embedding -> LSTM -> Linear."""

    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embed = self.embedding(x)
        out, _ = self.lstm(embed)
        out = self.fc(out[:, -1, :])
        return out


def build_lstm_model(vocab_size=1000, embed_size=64, hidden_size=128, output_size=2):
    return LSTMModel(vocab_size, embed_size, hidden_size, output_size)


# ---------------------------------------------------------------------------
# Transformer pipelines
# ---------------------------------------------------------------------------
_SENTIMENT_PIPE = None
_GENERATOR_PIPE = None


def get_sentiment_pipeline():
    global _SENTIMENT_PIPE
    if _SENTIMENT_PIPE is None:
        _SENTIMENT_PIPE = pipeline("sentiment-analysis")
    return _SENTIMENT_PIPE


def get_generator_pipeline():
    global _GENERATOR_PIPE
    if _GENERATOR_PIPE is None:
        _GENERATOR_PIPE = pipeline("text-generation", model="gpt2")
    return _GENERATOR_PIPE


def analyze_sentiment(text: str):
    return get_sentiment_pipeline()(text)


def generate_text(prompt: str, max_length: int = 50):
    return get_generator_pipeline()(prompt, max_length=max_length)[0]["generated_text"]
