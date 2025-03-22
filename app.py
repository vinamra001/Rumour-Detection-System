from flask import Flask, render_template, request, jsonify
import pickle
import re
import string
import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Load the trained model and TF-IDF vectorizer
with open("naive_bayes_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Google Fact Check API details (replace with your actual API key)
FACT_CHECK_API_KEY = "AIzaSyBGoW7Q8teenIn1FUaXE6jZSisLVgEx_7A"
FACT_CHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# Initialize Flask app
app = Flask(__name__)

# Optimized text preprocessing function (preserving key words)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = word_tokenize(text)
    english_stopwords = set(stopwords.words("english"))
    preserved_words = {"5g", "covid-19", "nasa", "rover", "perseverance"}
    filtered_words = [word for word in words if word not in english_stopwords or word in preserved_words]
    return " ".join(filtered_words)

# Function to query the Google Fact Check API
def fact_check_api(query):
    params = {"query": query, "key": FACT_CHECK_API_KEY}
    response = requests.get(FACT_CHECK_API_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        if "claims" in data and len(data["claims"]) > 0:
            fact_results = []
            for claim in data["claims"]:
                fact_results.append({
                    "text": claim["text"],
                    "claimant": claim.get("claimant", "Unknown"),
                    "rating": claim["claimReview"][0]["textualRating"]
                })
            return fact_results
    return [{"text": "No fact-check results found", "claimant": "N/A", "rating": "N/A"}]

# Home route: renders the input form
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Prediction route: processes input, queries the model and Fact Check API, and combines results
@app.route("/predict", methods=["POST"])
def predict():
    user_text = request.form["text"]
    cleaned_text = clean_text(user_text)
    transformed_text = tfidf_vectorizer.transform([cleaned_text])
    prediction = model.predict(transformed_text)
    model_result = "Rumor" if prediction[0] == 1 else "True News"
    
    # Get Fact Check API results
    fact_results = fact_check_api(user_text)
    if fact_results and fact_results[0]["rating"].lower() not in ["n/a"]:
        rating = fact_results[0]["rating"].lower()
    else:
        rating = "n/a"

    # Refined final decision logic:
    if rating in ["false", "misleading"]:
        final_decision = "Verified Rumor 🚫"
    elif rating in ["true", "mostly true"]:
        final_decision = "Verified True News ✅"
    elif rating == "n/a":
    # When no fact-check data is available, we mark for manual review
        final_decision = f"Model Prediction: {model_result} (Manual review recommended)"
    else:
        final_decision = f"Model Prediction: {model_result} (Further review needed)"
    
    return render_template("result.html",
                           input_text=user_text,
                           model_prediction=model_result,
                           fact_check_results=fact_results,
                           final_decision=final_decision)

if __name__ == "__main__":
    app.run(debug=True)
