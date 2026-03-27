# 🛍️ Product Review Sentiment Analysis using ML and NLP
## 📘 Overview

This project focuses on analyzing customer product reviews to determine sentiment — Positive, Neutral, or Negative — using Natural Language Processing (NLP) and Machine Learning techniques.

The goal is to transform unstructured textual feedback into meaningful insights that help businesses improve products, enhance customer experience, and make data-driven decisions.

## 🎯 Objectives
- Analyze customer reviews from e-commerce platforms
- Perform text preprocessing and feature engineering
- Build and compare multiple machine learning models
- Classify sentiment into Positive, Neutral, and Negative
- Generate actionable insights from customer feedback


## 🧠 Dataset
Dataset Name: Product Review Dataset
Source: E-commerce platforms (Amazon / etc.)
Features: Text reviews + metadata
Target Variable: Sentiment Label (Positive / Neutral / Negative)

## Key Attributes
- Review Text
- Rating
- Product Category
- Review Length
- Customer Feedback

## ⚙️ Workflow
### 1. Data Preprocessing
- Removed missing values
- Lowercasing text
- Removed punctuation & special characters
- Stopword removal
- Tokenization
- Custom preprocessing using TextCleaner

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
- SVM classification
- LightGBM 
- ExtraTree classification
