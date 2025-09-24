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



---

## 🚀 Features
* **Dataset**: [Netflix Facebook user comments](https://www.kaggle.com/datasets/tomthescientist/netflix-facebook-posts-as-sentences-for-llm-input)
* **Text Preprocessing**: lowercasing, URL/mention/hashtag handling, punctuation removal, stopword removal, lemmatization.
* **Sentiment Analysis**  
  * Baseline: VADER (rule-based).
  * Supervised: TF-IDF + Linear SVM (trained using VADER labels as weak supervision).
* **Visualizations**  
  * Sentiment distribution bar chart.
  * Sentiment trend over time (monthly line chart).
  * Wordclouds per sentiment class.
* **Topic Modeling**: Latent Dirichlet Allocation (LDA) to discover key discussion themes.
* **Interactive App**: Streamlit dashboard for predictions, dataset upload, wordclouds, topic modeling, and sentiment-over-time visualization.

---

## 🛠️ Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/kau-z/netflix-fb-sentiment.git
   cd netflix-fb-sentiment

2. **Create & activate a virtual environment**

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

3. **Install dependencies**

pip install -r requirements.txt

▶️ Run the Streamlit App
streamlit run src/app.py


Open the URL shown in the terminal (usually http://localhost:8501).

Features in the app:

Text box: Enter a comment and get an instant sentiment prediction.

CSV upload: Upload the Netflix FB dataset (or your own) to:

View sentiment distribution and monthly trend.

Generate sentiment-specific wordclouds.

Discover topics with LDA.


⚖️ Notes & Ethics

Data is from a public Kaggle dataset of Facebook comments—no private scraping.

If you gather additional Facebook data yourself, ensure you comply with Facebook/Meta terms and privacy policies.