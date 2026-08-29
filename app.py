"""
app.py
------
Streamlit demo app for the NLP Toolkit.

Run locally with:
    streamlit run app.py

Deploy on Streamlit Community Cloud by pointing it at this file
in your GitHub repo (see README.md for step-by-step instructions).
"""

import streamlit as st
import pandas as pd

from nlp_utils import (
    ensure_nltk_resources,
    tokenize,
    remove_stopwords,
    stem_words,
    lemmatize_text,
    bag_of_words,
    tfidf_vectorize,
    pos_tag_text,
    named_entities,
    train_naive_bayes,
    classification_metrics,
    train_word2vec,
    build_lstm_model,
    analyze_sentiment,
    generate_text,
)

st.set_page_config(page_title="NLP Toolkit", page_icon="🧠", layout="wide")

# Download NLTK data once per session (cached across reruns).
@st.cache_resource
def _setup():
    ensure_nltk_resources()
    return True


_setup()

st.title("🧠 NLP Toolkit — Interactive Demo")
st.caption(
    "A hands-on demo of classic + modern NLP techniques: tokenization, "
    "stemming/lemmatization, BoW/TF-IDF, POS tagging, NER, Naive Bayes "
    "classification, Word2Vec, an LSTM classifier, and Transformer pipelines."
)

tabs = st.tabs(
    [
        "1️⃣ Preprocessing",
        "2️⃣ Vectorization",
        "3️⃣ POS & NER",
        "4️⃣ Naive Bayes Classifier",
        "5️⃣ Word2Vec",
        "6️⃣ LSTM Model",
        "7️⃣ Transformers",
    ]
)

# ---------------------------------------------------------------------------
# 1. Preprocessing
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Tokenization, Stopword Removal, Stemming & Lemmatization")
    text = st.text_area(
        "Enter text",
        value="Dr. Smith loves programming. He often uses Python 3.9!",
        key="preprocess_text",
    )
    if st.button("Run preprocessing", key="btn_preprocess"):
        words, sentences = tokenize(text)
        filtered = remove_stopwords(words)
        stemmed = stem_words(filtered)
        lemmas = lemmatize_text(text)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Word tokens**")
            st.write(words)
            st.markdown("**Sentence tokens**")
            st.write(sentences)
            st.markdown("**After stopword removal**")
            st.write(filtered)
        with col2:
            st.markdown("**Stemmed words (Porter Stemmer)**")
            st.write(stemmed)
            st.markdown("**Lemmas (spaCy)**")
            st.write(lemmas)

# ---------------------------------------------------------------------------
# 2. Vectorization
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Bag-of-Words vs TF-IDF")
    corpus_text = st.text_area(
        "Enter one document per line",
        value="I love NLP\nNlp is very Powerful",
        key="corpus_text",
    )
    if st.button("Vectorize", key="btn_vectorize"):
        corpus = [line for line in corpus_text.split("\n") if line.strip()]
        bow_features, bow_matrix = bag_of_words(corpus)
        tfidf_features, tfidf_matrix = tfidf_vectorize(corpus)

        st.markdown("**Bag-of-Words**")
        st.dataframe(pd.DataFrame(bow_matrix, columns=bow_features))

        st.markdown("**TF-IDF**")
        st.dataframe(pd.DataFrame(tfidf_matrix, columns=tfidf_features).round(3))

# ---------------------------------------------------------------------------
# 3. POS & NER
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Part-of-Speech Tagging & Named Entity Recognition")
    ner_text = st.text_input(
        "Enter text",
        value="Google was founded in California by Larry Page and Sergey Brin",
        key="ner_text",
    )
    if st.button("Analyze", key="btn_ner"):
        tags = pos_tag_text(ner_text)
        entities = named_entities(ner_text)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**POS tags**")
            st.dataframe(pd.DataFrame(tags, columns=["Token", "POS Tag"]))
        with col2:
            st.markdown("**Named entities**")
            if entities:
                st.dataframe(pd.DataFrame(entities, columns=["Entity", "Label"]))
            else:
                st.write("No entities found.")

# ---------------------------------------------------------------------------
# 4. Naive Bayes classifier
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Naive Bayes Sentiment Classifier (toy example)")
    st.write(
        "Trains a `MultinomialNB` classifier on a tiny labeled dataset "
        "(1 = positive, 0 = negative) to demonstrate the classic "
        "text-classification pipeline."
    )
    default_texts = "I love this product\nthis is bad\namazing experience\nworst service"
    texts_input = st.text_area("Training texts (one per line)", value=default_texts)
    labels_input = st.text_input("Labels (comma-separated, matching line order)", value="1,0,1,0")

    if st.button("Train & Evaluate", key="btn_nb"):
        texts = [t for t in texts_input.split("\n") if t.strip()]
        labels = [int(x) for x in labels_input.split(",")]
        if len(texts) != len(labels):
            st.error("Number of texts and labels must match.")
        else:
            model, vectorizer, y_test, y_pred = train_naive_bayes(texts, labels)
            st.write("Predictions on held-out test split:", list(y_pred))
            if len(set(y_test)) > 1:
                metrics = classification_metrics(y_test, y_pred)
                st.json(metrics)
            else:
                st.info("Test split too small/uniform for full metrics — add more rows.")

# ---------------------------------------------------------------------------
# 5. Word2Vec
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Word2Vec Embeddings (gensim)")
    st.write("Trains a tiny Word2Vec model on sample tokenized sentences.")
    if st.button("Train Word2Vec", key="btn_w2v"):
        sentences = [
            ["natural", "language", "processing"],
            ["machine", "learning", "ai"],
        ]
        w2v_model = train_word2vec(sentences)
        word = "natural"
        st.write(f"Vector for '{word}' (first 10 dims):")
        st.write(w2v_model.wv[word][:10])

# ---------------------------------------------------------------------------
# 6. LSTM model
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("LSTM Classifier Architecture (PyTorch)")
    st.write(
        "Builds an untrained `Embedding -> LSTM -> Linear` classifier to "
        "show the architecture used for sequence classification tasks."
    )
    if st.button("Build LSTM model", key="btn_lstm"):
        model = build_lstm_model()
        st.code(str(model), language="text")
        st.success("LSTM model instantiated (weights are randomly initialized).")

# ---------------------------------------------------------------------------
# 7. Transformer pipelines
# ---------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Pretrained Transformer Pipelines (Hugging Face)")
    st.warning(
        "These download pretrained model weights the first time they run, "
        "which can be slow/heavy on free hosting tiers."
    )

    st.markdown("**Sentiment analysis**")
    sent_text = st.text_input("Text to analyze", value="I really enjoy learning NLP")
    if st.button("Analyze sentiment", key="btn_sentiment"):
        with st.spinner("Loading sentiment model..."):
            result = analyze_sentiment(sent_text)
        st.json(result)

    st.markdown("**Text generation (GPT-2)**")
    gen_prompt = st.text_input("Prompt", value="NLP is the future because")
    if st.button("Generate text", key="btn_generate"):
        with st.spinner("Loading GPT-2 and generating..."):
            generated = generate_text(gen_prompt, max_length=50)
        st.write(generated)

st.divider()
st.caption("Built with NLTK, spaCy, scikit-learn, gensim, PyTorch, and Hugging Face Transformers.")
