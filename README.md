# 🎬 Netflix Facebook Comments – Sentiment Analysis & NLP Dashboard

An end-to-end Natural Language Processing (NLP) project that analyzes real Facebook comments about Netflix.  
It includes text preprocessing, sentiment classification, wordcloud visualizations, topic modeling (LDA),  
and an interactive **Streamlit** dashboard.

---

## 📂 Project Structure
netflix-fb-sentiment/
├─ src/
│ ├─ app.py # Streamlit dashboard
│ ├─ predict.py # CLI predictor (optional)
│ └─ models/ # Saved TF-IDF & SVM model (if trained)
├─ notebooks/
│ └─ fb_nlp.ipynb # Data exploration & model training
├─ data/ # (keep small samples only, don’t commit full dataset)
├─ requirements.txt
└─ README.md