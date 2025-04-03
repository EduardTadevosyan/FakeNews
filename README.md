# Fake News Detection Using Logistic Regression in Python

This project focuses on detecting fake news articles using natural language processing (NLP) and logistic regression. The goal is to classify whether a given news article is real or fake based on its content — a crucial task in today’s digital world where misinformation spreads rapidly.

The model is trained on a labeled dataset of real and fake news articles and achieves high accuracy using a clean and effective machine learning pipeline.

🧠 **Model**: Logistic Regression  
🗂️ **Code**: [Jupyter Notebook](https://github.com/EduardTadevosyan/FakeNews/blob/main/FakenewsProject.ipynb)  
📄 **Dataset**: [Kaggle – Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## 🚀 Objective

The objective of this project is to build a machine learning model that can distinguish fake news from real news using only the text of the articles. It demonstrates how classical NLP techniques like TF-IDF and logistic regression can solve a real-world classification problem effectively.

---

## 🛠️ Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- WordCloud

---

## 🔧 How It Works

1. **Data Loading & Cleaning**
   - Merged real and fake news datasets
   - Removed nulls and unnecessary columns
   - Preprocessed the text (lowercasing, punctuation removal, stopword filtering)

2. **Text Vectorization**
   - Used TF-IDF to convert textual data into numerical features

3. **Model Training**
   - Trained a Logistic Regression model using Scikit-learn
   - Evaluated the model with accuracy score, classification report, and confusion matrix

---

## 📈 Results

The model performed with ~99% accuracy on the test set. Below is the classification report summary and the accuracy curve:

![Result](https://github.com/EduardTadevosyan/FakeNews/blob/main/Images/accuracy.png)



