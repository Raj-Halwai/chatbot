# train.py
import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nlp_utils import preprocess

# Load training data
with open('data/intents.json', 'r') as f:
    intents = json.load(f)

texts = []
labels = []

# Properly preprocess patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        texts.append(preprocess(pattern))  # ✅ PREPROCESS inside loop
        labels.append(intent['tag'])

# Convert text to numeric features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train classifier
model = LogisticRegression()
model.fit(X, labels)

# Save model and vectorizer
with open('model/intent_model.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)

print("✅ Model trained and saved as 'intent_model.pkl'")
