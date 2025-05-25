# **Spam Detection App using Machine Learning**

This project is a machine learning-based spam detection system using **Logistic Regression** and **TF-IDF** for feature extraction. It is designed to classify messages into two categories: **Spam** or **Not Spam**.

## **Table of Contents**
- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Model and Data](#model-and-data)
- [Contributing](#contributing)
- [License](#license)

---

## **Introduction**

Spam messages are unwanted or unsolicited messages, often sent in bulk. The objective of this project is to develop a machine learning model that can classify a given message as **Spam** or **Not Spam** based on its content.

This app uses **Logistic Regression** as the classifier and **TF-IDF (Term Frequency-Inverse Document Frequency)** for transforming the text into numerical features. The trained model can be accessed through a simple Flask app for local deployment.

---

## **Technologies Used**
- **Flask**: Web framework for creating the web application.
- **Scikit-learn**: Machine learning library used for training and predicting.
- **Joblib**: For saving and loading the model and vectorizer.
- **NLTK (Natural Language Toolkit)**: For text preprocessing like stopwords removal.
- **NumPy**: Array manipulation (indirect dependency).
- **TF-IDF Vectorizer**: For converting text data into numerical form.

---

## **Setup and Installation**

Follow these steps to run the project on your local machine:

### 1. **Clone the repository**

Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/spam-detection-app.git
cd spam-detection-app
