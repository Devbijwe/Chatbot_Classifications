from flask import Flask, render_template, request, jsonify
import json

import random
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV

app = Flask(__name__)

# Load intents from the JSON file
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Initialize chatbot components (vectorizer and classifier)
vectorizer = TfidfVectorizer(max_df=0.85, max_features=1000, stop_words='english')
clf = LogisticRegression(random_state=0, max_iter=1000, C=1.0, solver='lbfgs')
calibrated_clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid')

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
calibrated_clf.fit(x, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']
    response = chatbot(user_input)
    return jsonify({"response": response})

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    intent_probs = calibrated_clf.predict_proba(input_text).max()

    # Define a confidence threshold
    confidence_threshold = 0.047  # Adjust as needed
    print(intent_probs)
    if intent_probs >= confidence_threshold:
        tag = calibrated_clf.predict(input_text)[0]
        for intent in intents:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                return response
    else:
        return "I'm sorry, I couldn't understand that. Please try rephrasing your question or provide more context."


