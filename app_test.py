from flask import Flask, request, jsonify
import pickle
import random
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Initialize and fit the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": [
            "Hi there",
            "Hello",
            "Hey",
            "I'm fine, thank you",
            "Nothing much",
        ],
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"],
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"],
    },
    {
        "tag": "about",
        "patterns": [
            "What can you do",
            "Who are you",
            "What are you",
            "What is your purpose",
        ],
        "responses": [
            "I am a chatbot",
            "My purpose is to assist you",
            "I can answer questions and provide assistance",
        ],
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": [
            "Sure, what do you need help with?",
            "I'm here to help. What's the problem?",
            "How can I assist you?",
        ],
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": [
            "I don't have an age. I'm a chatbot.",
            "I was just born in the digital world.",
            "Age is just a number for me.",
        ],
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": [
            "I'm sorry, I cannot provide real-time weather information.",
            "You can check the weather on a weather app or website.",
        ],
    },
]

corpus = [pattern for intent in intents for pattern in intent['patterns']]
vectorizer.fit(corpus)

# Transform the patterns and train the classifier
X = vectorizer.transform(corpus)
y = [intent['tag'] for intent in intents for _ in intent['patterns']]
classifier = LogisticRegression(random_state=0, max_iter=10000)
classifier.fit(X, y)

# Save the fitted vectorizer and classifier
filename = "mdl_chat.pkl"
pickle.dump((vectorizer, classifier), open(filename, "wb"))

# Load the trained model
loaded_vectorizer, loaded_model = pickle.load(open(filename, "rb"))

# Function to get chatbot response
def chatbot_response(text):
    input_text = loaded_vectorizer.transform([text])
    predicted_tag = loaded_model.predict(input_text)[0]
    for i in intents:
        if i["tag"] == predicted_tag:
            response = random.choice(i["responses"])
            return response

# Initialize the RoBERTa classifier
def roberta_classifier(query):
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    model_outputs = classifier(query)
    data = model_outputs[0][0]
    label_value = data["label"]
    score_value = data["score"]
    return label_value, score_value

# Process the query for emotion and depression analysis
def process_query(query, sum, cnt, results):
    label, score = roberta_classifier(str(query))
    quotient = float(score) * 100
    sum += quotient
    results.append([label, quotient])
    cnt += 1
    return sum, cnt, results

# Load the Keras model and tokenizer
model_form = load_model("model.keras")
token_form = pickle.load(open('tokenizer.pkl', 'rb'))

def depression_measure(query, predicted_value, counter):
    twt = token_form.texts_to_sequences(query)
    twt = pad_sequences(twt, maxlen=50)
    prediction = model_form.predict(twt)[0][0]
    predicted_value += prediction
    counter += 1
    return predicted_value, counter

# Create Flask application
app = Flask(__name__)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    query = data['query']
    
    output = chatbot_response(query)
    
    global sum, cnt, results, predicted_value, counter
    sum, cnt, results = process_query(query, sum, cnt, results)
    predicted_value, counter = depression_measure([query], predicted_value, counter)
    
    response = {
        "response": output,
        "emotion_analysis": results,
        "depression_level": predicted_value / counter if counter > 0 else 0
    }
    
    return jsonify(response)

if __name__ == '__main__':
    sum = 0
    cnt = 0
    results = []
    predicted_value = 0
    counter = 0
    app.run(debug=True)
