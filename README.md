# 🛍️ Product Review Sentiment Analysis using ML and NLP
## 📘 Overview

This project focuses on **analyzing customer product reviews** to determine sentiment — **Positive, Neutral, or Negative** — using **Natural Language Processing (NLP) and Machine Learning techniques.**

The goal is to transform unstructured textual feedback into meaningful insights that help businesses improve products, enhance customer experience, and make data-driven decisions.

---

## 🎯 Business Problem

E-commerce platforms receive millions of reviews daily, making manual analysis impractical.

### ❗ Challenges:
- Unstructured text data
- Large-scale volume
- Delayed decision-making
### ✅ Solution:

This system automates sentiment classification to:

- Extract insights in real-time
- Identify customer pain points
- Improve product quality & user experience

---

## 🚀 Key Features

✔ End-to-End ML Pipeline

✔ Custom Text Cleaning (TextCleaner)

✔ TF-IDF with N-grams

✔ Hyperparameter Tuning (GridSearchCV)

✔ Model Comparison Dashboard

✔ Streamlit Interactive UI

✔ WordCloud Visualization

---

## 🧠 Dataset
**Dataset Name:** Product Review Dataset

**File:** Amazon Product Review.txt

**Features:** Text reviews + metadata

**Target Variable:** Sentiment Label (Positive / Neutral / Negative)

### Key Attributes
- Review Text
- Rating
- Product Category
- Review Length
- Customer Feedback

---

## ⚙️ Workflow

### 1. Data Preprocessing
- Removed missing values
- Lowercasing text
- Removed punctuation & special characters
- Stopword removal
- Tokenization
- Custom preprocessing using **TextCleaner**

### 2. Exploratory Data Analysis (EDA)
- Sentiment distribution
- Word frequency analysis
- WordCloud visualization
- Review length distribution

### 3. Feature Engineering
- TF-IDF Vectorization
- N-grams (Unigrams & Bigrams)
- Text normalization

### 4. Model Development
**Implemented and compared:**
- Logistic Regression
- Multinomial Naive Bayes
- LinearSVM classification
- LightGBM 
- ExtraTree classification

### 5. Hyperparameter Tuning

**Used GridSearchCV for optimization:**
- Logistic Regression → C, solver
- Multinomial Naive Bayes → alpha
- LinearSVM classification
- LightGBM
- ExtraTree classification

---

## 📊 Model Performance Summary

| Model                   | Best Parameters                | Accuracy | Precision | Recall | F1-Score |
| ----------------------- | ------------------------------ | -------- | --------- | ------ | -------- |
| **Logistic Regression** | C=1, solver=lbfgs              | **89%**  | 88%       | 89%    | **Best** |
| **LinearSVM**           | n_estimators=200, max_depth=20 | 87%      | 86%       | 87%    | High     |
| **Naive Bayes**         | alpha=1.0                      | 85%      | 84%       | 85%    | Moderate |
| **LightGBM**            | alpha=1.0                      | 85%      | 84%       | 85%    | Moderate |

**🏆 Best Model:** Logistic Regression

---

## 📈 Sample Predictions

| Review                               | Predicted Sentiment |
| ------------------------------------ | ------------------- |
| "Amazing product, highly recommend!" | Positive ✅          |
| "It's okay, nothing special."        | Neutral 😐          |
| "Very bad quality, waste of money."  | Negative ❌          |

---

## 💻 Streamlit UI

**Run:**  streamlit run streamlit_app.py

**Features:**
Text input for review
Instant prediction
WordCloud visualization

---

## 💡 Key Insights

- Most reviews tend to be **positive**, indicating overall customer satisfaction
- Negative reviews highlight **product quality and delivery issues**
- TF-IDF + Logistic Regression works best for text classification
- Short reviews are harder to classify compared to detailed reviews

---

## ⚠️ Limitations

- Model performance depends on text quality
- Sarcasm and context are difficult to detect
- Limited dataset may affect generalization

---

## 🚀 Future Enhancements

- Implement **Deep Learning (LSTM / BERT)**
- Perform **Aspect-Based Sentiment Analysis**
- Add **real-time review scraping**

---

## 🧩 Tools & Technologies

- **Language:** Python
- **Libraries:** pandas, numpy, scikit-learn, nltk
- **Visualization:** matplotlib, seaborn, wordcloud
- **Frontend:** Streamlit
- **Environment:** Jupyter Notebook / VS Code

---

## 🌐 Deployment

- Interactive UI using **Streamlit**
- Real-time sentiment prediction system

---

## 📅 Project Completion Date: MAR 2026

---

## 👨‍💻 Author
R. Nitin
📧 **Email**: rnitin0604@gmail.com

🔗 **LinkedIn:** https://www.linkedin.com/in/nitin0604/

---

## 🩹 License

This project is licensed under the **MIT License** — free to use for educational and research purposes.



