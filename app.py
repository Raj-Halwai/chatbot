from flask import Flask, request, jsonify
from flask_cors import CORS
import random, json, pickle
from nlp_utils import preprocess

app = Flask(__name__)
CORS(app)

# Load model
with open('model/intent_model.pkl', 'rb') as f:
    vectorizer, model = pickle.load(f)

# Load intents
with open('data/intents.json') as f:
    intents = json.load(f)

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get("message")
    if not user_msg:
        return jsonify({"response": "Please enter a message."}), 400

    processed_input = preprocess(user_msg)  # ✅ preprocess before prediction
    X = vectorizer.transform([processed_input])
    predicted_intent = model.predict(X)[0]

    for intent in intents['intents']:
        if intent['tag'] == predicted_intent:
            return jsonify({"response": random.choice(intent['responses'])})

    return jsonify({"response": "Sorry, I didn't understand that."})

if __name__ == '__main__':
    app.run(debug=True)
