import streamlit as st
import pandas as pd
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re, string

nltk.download('stopwords')
nltk.download('wordnet')

st.set_page_config(page_title="Netflix FB Sentiment", page_icon="🎬")
st.title("🎬 Netflix Facebook Comments — Sentiment Dashboard")

# ---------- Load trained classifier (if you created one) ----------
@st.cache_resource
def load_model():
    try:
        vec = joblib.load("src/models/tfidf.joblib")
        clf = joblib.load("src/models/linearsvc.joblib")
        return vec, clf
    except:
        return None, None

vectorizer, classifier = load_model()
analyzer = SentimentIntensityAnalyzer()

# ---------- Text-box prediction ----------
st.header("🔎 Quick Sentiment Check")
text = st.text_area("Type a comment:", "I love this new Netflix series!")

def predict_sentiment(txt):
    if vectorizer and classifier:
        X = vectorizer.transform([txt])
        return classifier.predict(X)[0]
    else:
        c = analyzer.polarity_scores(txt)['compound']
        return 'positive' if c >= 0.05 else ('negative' if c <= -0.05 else 'neutral')

if st.button("Analyze"):
    st.subheader(f"Prediction: **{predict_sentiment(text)}**")

# ---------- Upload full dataset ----------
st.header("📂 Explore a Netflix FB Comments CSV")
uploaded = st.file_uploader("Upload CSV (needs a 'Comment' column)", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)

    # Normalise column names
    if "Comment" in df.columns:
        df = df.rename(columns={'Comment': 'text'})
    if "text" not in df.columns:
        st.error("CSV must contain a 'Comment' or 'text' column.")
    else:
        st.write("Sample rows:", df.head())

        # --- Preprocess text
        sw  = set(stopwords.words('english'))
        lem = WordNetLemmatizer()
        def preprocess(s):
            s = str(s).lower()
            s = re.sub(r"http\\S+|www\\S+", " url ", s)
            s = re.sub(r"@[A-Za-z0-9_]+", " user ", s)
            s = re.sub(r"#(\\w+)", r"\\1", s)
            s = s.translate(str.maketrans("", "", string.punctuation))
            return " ".join(lem.lemmatize(t) for t in s.split() if t not in sw)
        df["clean"] = df["text"].apply(preprocess)

        # --- VADER sentiment for each comment
        def to_lab(c): 
            return 'positive' if c>=0.05 else ('negative' if c<=-0.05 else 'neutral')
        df["compound"]    = df["text"].apply(lambda t: analyzer.polarity_scores(str(t))["compound"])
        df["vader_label"] = df["compound"].apply(to_lab)

        st.subheader("Sentiment distribution")
        st.bar_chart(df["vader_label"].value_counts())

        # --- Sentiment over time (if Timestamp present)
        if "Timestamp" in df.columns:
            st.subheader("📈 Sentiment trend over time")
            # parse dates safely
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            ts = (df.dropna(subset=["Timestamp"])
                    .groupby([pd.Grouper(key="Timestamp", freq="M"), "vader_label"])
                    .size()
                    .unstack(fill_value=0))
            st.line_chart(ts)

        # --- Wordclouds
        st.subheader("☁️ Wordcloud by Sentiment")
        for lab in sorted(df["vader_label"].unique()):
            text_blob = " ".join(df.loc[df["vader_label"]==lab,"clean"])
            if not text_blob.strip(): continue
            wc = WordCloud(width=800, height=400, background_color="white").generate(text_blob)
            fig, ax = plt.subplots(figsize=(8,4))
            ax.imshow(wc); ax.axis("off")
            st.pyplot(fig)

        # --- Topic modelling (LDA)
        st.subheader("🔍 Topics (LDA)")
        tokens = [[w for w in t.split()] for t in df["clean"].dropna()]
        dictionary = corpora.Dictionary(tokens)
        dictionary.filter_extremes(no_below=5, no_above=0.5)
        corpus = [dictionary.doc2bow(t) for t in tokens]
        if len(dictionary) and len(corpus):
            lda = LdaModel(corpus=corpus, id2word=dictionary,
                           num_topics=3, passes=5, random_state=42)
            for i in range(3):
                st.write(f"**Topic {i}:** {lda.print_topic(i, topn=8)}")
        else:
            st.info("Not enough text for LDA.")
