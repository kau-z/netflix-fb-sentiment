import sys, joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def predict_text(text: str) -> str:
    try:
        vec = joblib.load("src/models/tfidf.joblib")
        clf = joblib.load("src/models/linearsvc.joblib")
        return clf.predict(vec.transform([text]))[0]
    except Exception:
        an = SentimentIntensityAnalyzer()
        c = an.polarity_scores(text)['compound']
        return 'positive' if c >= 0.05 else ('negative' if c <= -0.05 else 'neutral')

if __name__ == "__main__":
    print(predict_text(" ".join(sys.argv[1:]) or "I love this!"))
