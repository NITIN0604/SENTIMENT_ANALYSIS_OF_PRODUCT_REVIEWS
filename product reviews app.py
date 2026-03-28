import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from sklearn.base import BaseEstimator, TransformerMixin

# ================================
# 🔥 CUSTOM TRANSFORMER (FIX ERROR)
# ================================
class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cleaned = []
        for text in X:
            text = str(text).lower()
            text = re.sub(r'<.*?>', '', text)
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            cleaned.append(text)
        return cleaned


# ================================
# LOAD MODEL
# ================================
model = joblib.load(r"E:\Product-Review-Sentiment-Analysis-main\Product-Review-Sentiment-Analysis-main\logistic_pipeline.pkl")

# ================================
# TITLE
# ================================
st.title("🌟 Product Review Sentiment Analysis")

# ================================
# USER INPUT
# ================================
st.header("✍️ Input Your Review")
user_review = st.text_area("Enter your product review:", "")

# ================================
# PREDICTION
# ================================
if st.button("Predict Sentiment"):
    if user_review:
        
        input_df = pd.DataFrame({'full_review': [user_review]})
        
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        # ✅ FIXED (STRING LABELS)
        sentiment_map = {
            'negative': "❌ Negative",
            'neutral': "⚖️ Neutral",
            'positive': "✅ Positive"
        }

        st.success(f"Prediction: **{sentiment_map[prediction]}**")

        # ✅ FIXED (dynamic class labels)
        st.subheader("Prediction Confidence")
        prob_df = pd.DataFrame({
            "Sentiment": model.classes_,
            "Probability": probabilities
        })

        st.bar_chart(prob_df.set_index("Sentiment"))
        
    else:
        st.warning("Please enter a review before predicting.")


# ================================
# LOAD DATA
# ================================
@st.cache_data
def load_data():
    return pd.read_csv(r"E:\Product-Review-Sentiment-Analysis-main\Product-Review-Sentiment-Analysis-main\processed_data.csv")

data = load_data()

# ================================
# SENTIMENT DISTRIBUTION
# ================================
st.header("📊 Sentiment Distribution")

# ✅ FIX (handle both cases safely)
if data['sentiment_label'].dtype == 'object':
    sentiment_count = data['sentiment_label'].value_counts()
    labels = sentiment_count.index
else:
    sentiment_count = data['sentiment_label'].value_counts().sort_index()
    labels = ["Negative", "Neutral", "Positive"]

plt.figure()
plt.bar(labels, sentiment_count.values)

plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.title("Sentiment Distribution")

st.pyplot(plt)

# ================================
# WORD CLOUD
# ================================
st.header("☁️ Word Cloud (All Reviews)")

text_data = " ".join(data['full_review'].astype(str))

wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text_data)

plt.figure()
plt.imshow(wordcloud)
plt.axis('off')

st.pyplot(plt)