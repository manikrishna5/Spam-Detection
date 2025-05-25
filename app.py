from flask import Flask, request, jsonify, render_template
import joblib
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("lr_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = text.replace("Subject", "")
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in text.split() if word.lower() not in stop_words]
    return " ".join(words)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_text = data.get("text", "")
    cleaned = clean_text(user_text)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)