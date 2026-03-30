import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from sklearn.base import BaseEstimator, TransformerMixin





# ------------------------------  PAGE CONFIG  ------------------------------

st.set_page_config(page_title="Sentiment App", layout="wide")



# -----------------------------  TEXT CLEANER  ------------------------------


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

# ------------------------------  LOAD MODEL  ------------------------------

model = joblib.load(r"E:\Sentiment Analysis of Product Reviews\logistic_pipeline.pkl")



# ------------------------------  LOAD DATA  ------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(r"E:\Sentiment Analysis of Product Reviews\cleaned_data.csv")

data = load_data()



# ------------------------------  SIDEBAR  ------------------------------

st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["Sentiment Prediction", "Data Insights"])



# ------------------------------  TITLE  ------------------------------

st.markdown(
    "<h1 style='text-align:center;'>🌟 Product Review Sentiment Analysis</h1>",
    unsafe_allow_html=True
)




# ----------------------------- PAGE 1: SENTIMENT PREDICTION -----------------------------

if page == "Sentiment Prediction":

    st.subheader("✍️ Enter Your Review")

    user_review = st.text_area("Enter your review:")

    if st.button("🚀 Predict Sentiment"):

        if user_review:

            input_df = pd.DataFrame({'full_review': [user_review]})

            with st.spinner("Analyzing sentiment..."):
                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]

            sentiment_map = {
                'negative': "❌ Negative",
                'neutral': "⚖️ Neutral",
                'positive': "✅ Positive"
            }

            st.markdown(f"""
            <div style="
                padding:20px;
                border-radius:10px;
                background-color:#1f77b4;
                color:white;
                text-align:center;
                font-size:24px;">
                Prediction: {sentiment_map.get(prediction, prediction)}
            </div>
            """, unsafe_allow_html=True)

            # -------- Gauge Meter --------

            st.subheader("🎯 Sentiment Score")

            sentiment_score = {
                'negative': 0,
                'neutral': 50,
                'positive': 100
            }

            score = sentiment_score.get(prediction, 50)
            st.progress(score / 100)
            st.markdown(f"**Score:** {score}/100")

            # -------- Feedback --------

            if prediction == 'positive':
                st.balloons()
                st.success("🎉 Great! Positive sentiment detected")
            elif prediction == 'negative':
                st.error("⚠️ Negative sentiment detected")
                st.toast("❌ Poor review detected!", icon="⚠️")
            else:
                st.info("⚖️ Neutral sentiment")
                st.toast("😐 Neutral review", icon="ℹ️")

            # -------- Probability Chart --------

            st.subheader("📊 Prediction Confidence")

            prob_df = pd.DataFrame({
                "Sentiment": model.classes_,
                "Probability": probabilities
            })

            st.bar_chart(prob_df.set_index("Sentiment"))

        else:
            st.warning("Please enter a review")



# ----------------------------- PAGE 2: DATA INSIGHTS -----------------------------

elif page == "Data Insights":

    st.subheader("📊 Dataset Overview")

    # -------- Premium Styled Note --------

    st.markdown("""
    <div style="
        padding:15px;
        border-radius:10px;
        background-color:#0e1117;
        border-left:5px solid #1f77b4;
        color:#FAFAFA;
        font-size:14px;">
        ℹ️ <b>Note:</b> The metrics displayed below represent the original distribution of the dataset. 
        These values are derived directly from the raw data and remain unchanged during the model development process.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------- Metrics --------

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Reviews", len(data))
    col2.metric("Positive %", round((data['sentiment_label']=='positive').mean()*100,2))
    col3.metric("Neutral %", round((data['sentiment_label']=='neutral').mean()*100,2))
    st.metric("Negative %", round((data['sentiment_label']=='negative').mean()*100,2))

    st.divider()


    # -------- Filter --------

    st.subheader("🔍 Filter Reviews")

    option = st.selectbox("Select Sentiment", ["All", "Positive", "Neutral", "Negative"])

    if option == "Positive":
        filtered = data[data['sentiment_label'] == 'positive']
    elif option == "Neutral":
        filtered = data[data['sentiment_label'] == 'neutral']
    elif option == "Negative":
        filtered = data[data['sentiment_label'] == 'negative']
    else:
        filtered = data

    st.dataframe(filtered.head(20))

    st.divider()

    # -------- Word Cloud --------

    st.subheader("☁️ Word Cloud")

    wc_option = st.radio("Choose Type", ["All", "Positive", "Neutral", "Negative"])

    if wc_option == "Positive":
        text_data = " ".join(data[data['sentiment_label']=='positive']['full_review'].astype(str))
    elif wc_option == "Neutral":
        text_data = " ".join(data[data['sentiment_label']=='neutral']['full_review'].astype(str))
    elif wc_option == "Negative":
        text_data = " ".join(data[data['sentiment_label']=='negative']['full_review'].astype(str))
    else:
        text_data = " ".join(data['full_review'].astype(str))

    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text_data)

    plt.figure()
    plt.imshow(wordcloud)
    plt.axis('off')

    st.pyplot(plt)



### -----------------------------  END OF APP  -----------------------------