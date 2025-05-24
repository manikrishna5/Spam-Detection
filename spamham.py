import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from wordcloud import WordCloud

df = pd.read_csv(r"C:\\Users\\Mani Krishna Karri\\Downloads\\spam_ham_dataset.csv")
ham_msg = df[df['label']=='ham']
spam_msg = df[df['label']=='spam']
ham_msg = ham_msg.sample(n=len(spam_msg),random_state=42)
bal_data = pd.concat([ham_msg,spam_msg]).reset_index(drop=True)

bal_data['text'] = bal_data['text'].astype(str).str.replace('Subject','')
pun_list = string.punctuation
def remove_punc(text):
    temp = str.maketrans('','',pun_list)
    return text.translate(temp)
bal_data['text'] = bal_data['text'].apply(lambda x: remove_punc(x))
def remove_stopwords(text):
    stop_words = stopwords.words('english')
    imp_words = []
    for word in str(text).split():
        word = word.lower()
        if word not in stop_words:
            imp_words.append(word)
    output = " ".join(imp_words)
    return output
bal_data['text'] = bal_data['text'].apply(lambda x: remove_stopwords(x))
train_X, test_X, train_Y, test_Y = train_test_split(
    bal_data['text'], bal_data['label'], test_size=0.2, random_state=42
)

tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
train_X_vect = tfidf.fit_transform(train_X)
test_X_vect = tfidf.transform(test_X)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(train_X_vect, train_Y)

# # model = MultinomialNB()
# # model.fit(train_X_vect, train_Y)

pred_Y = model.predict(test_X_vect)

# print("📊 Accuracy:", accuracy_score(test_Y, pred_Y))
# print("\n📄 Classification Report:\n", classification_report(test_Y, pred_Y))
# print("\n🧮 Confusion Matrix:\n", confusion_matrix(test_Y, pred_Y))
import joblib

joblib.dump(model, "lr_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")






    
