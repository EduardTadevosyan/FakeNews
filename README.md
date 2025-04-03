# Fake News Detection Using LSTM in Python 📰🔍

This project focuses on identifying fake news articles using natural language processing and a deep learning model. By training an LSTM (Long Short-Term Memory) neural network on a labeled dataset of real and fake news, the model is able to classify news articles with high accuracy.

🧠 **Model**: LSTM Neural Network  
📊 **Accuracy**: ~97% on test data  
📄 **Dataset**: [Kaggle – True and Fake News](https://www.kaggle.com/code/yossefmohammed/true-and-fake-news-lstm-accuracy-97-90)

---

## 🚀 Objective

The goal of this project was to build a binary classification model that can distinguish between real and fake news articles based solely on the text content (title and article body). It demonstrates how deep learning techniques can be applied to real-world problems like misinformation detection.

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- LSTM (RNN)
- Pandas, NumPy
- Natural Language Toolkit (NLTK)
- Matplotlib / Seaborn

---

## 🔧 How It Works

1. **Text Preprocessing**
   - Lowercasing, punctuation removal, stopword filtering
   - Tokenization and padding of input sequences

2. **Model Architecture**
   - Embedding layer
   - LSTM layer
   - Dense output layer with sigmoid activation

3. **Training**
   - Binary cross-entropy loss
   - Validation on test split
   - Achieved ~97% accuracy

---

## 📈 Results

- **Accuracy**: ~97%  
- **Loss curve & Accuracy curve** plotted to monitor overfitting  
- Final model can predict whether an unseen article is **Fake** or **Real**

---

## 📂 Project Structure

