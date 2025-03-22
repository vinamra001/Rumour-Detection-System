import os
import requests
import pickle
import nltk
import re
import string
from flask import Flask, request, render_template
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Load stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def check_fact_with_google(query):
    api_key = os.getenv('AIzaSyBGoW7Q8teenIn1FUaXE6jZSisLVgEx_7A')  # Use environment variable
    if not api_key:
        return "Error: Missing API Key"
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'claims' in data:
            return data['claims'][0]['text']
        else:
            return "No fact check found."
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['text']
    processed_input = preprocess_text(user_input)
    input_tfidf = tfidf_vectorizer.transform([processed_input])
    prediction = model.predict(input_tfidf)[0]
    fact_check_result = check_fact_with_google(user_input)
    result = "True News" if prediction == 1 else "Rumor"
    return render_template('result.html', prediction=result, fact_check=fact_check_result)

if __name__ == "__main__":
    app.run(debug=True)
