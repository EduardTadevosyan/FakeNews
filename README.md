# Fake News Detection Using Logistic Regression in Python

This project focuses on detecting fake news articles using natural language processing (NLP) and logistic regression. The goal is to classify whether a given news article is real or fake based on its content. This type of classification is increasingly important in today’s digital world, where misinformation spreads rapidly online.

The model is trained on a labeled dataset of real and fake news articles and achieves high accuracy using a simple but effective machine learning pipeline.

🧠 **Model**: Logistic Regression  
🗂️ **Code**: [Jupyter Notebook](https://github.com/EduardTadevosyan/FakeNews/blob/main/FakenewsProject.ipynb)  
📄 **Dataset**: [Kaggle – Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)


---

## 🚀 Objective

The objective of this project is to build a machine learning model that can distinguish fake news from real news using only the text of the articles. This project demonstrates the use of classical NLP techniques like TF-IDF and logistic regression to solve a real-world classification problem.

---

## 🛠️ Libraries Used

- Python
- WordCloud
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Matplotlib
- Seaborn (for visualizations)

---

## 🔧 How It Works

1. **Data Loading and Cleaning**
   - Combined fake and real news datasets
   - Removed nulls and unnecessary columns
   - Preprocessed the text (lowercasing, stopwords, punctuation removal)

2. **Text Vectorization**
   - Applied TF-IDF to convert text into numerical feature vectors

3. **Model Training**
   - Trained a Logistic Regression model on the TF-IDF features
   - Evaluated using accuracy, classification report, and confusion matrix

---

## 📈 Results

- **Accuracy**: ~99%  
- **Evaluation**:
  - Precision, Recall, F1-Score reported via `classification_report`
  - Confusion matrix plotted for visual inspection

Example output:
          precision    recall  f1-score   support

       0       0.98      0.99      0.99      4221
       1       0.99      0.98      0.99      3432

accuracy                           0.99      7653



