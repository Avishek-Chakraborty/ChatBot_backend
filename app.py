from flask import Flask, request, jsonify, render_template
import pickle
import random
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from intent import intents

# Initialize and fit the TF-IDF vectorizer
vectorizer = TfidfVectorizer()


corpus = [pattern for intent in intents for pattern in intent["patterns"]]
vectorizer.fit(corpus)

# Transform the patterns and train the classifier
X = vectorizer.transform(corpus)
y = [intent["tag"] for intent in intents for _ in intent["patterns"]]
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
    classifier = pipeline(
        task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None
    )
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
    results.append({"query": query, "emotion": label, "score": quotient})
    cnt += 1
    return sum, cnt, results


# Load the Keras model and tokenizer
model_form = load_model("model.keras")
token_form = pickle.load(open("tokenizer.pkl", "rb"))


def depression_measure(query, predicted_value, counter):
    twt = token_form.texts_to_sequences(query)
    twt = pad_sequences(twt, maxlen=50)
    prediction = model_form.predict(twt)[0][0]
    predicted_value += prediction
    counter += 1
    return predicted_value, counter


# Calculate the severity levels for depression
def severity_levels(predicted_value, counter):
    thresholds = [0.2, 0.4, 0.6, 0.8]
    predicted_value = round((predicted_value / counter) * 10, 5)

    if predicted_value < thresholds[0]:
        severity = "No depression"
    elif predicted_value < thresholds[1]:
        severity = "Mild depression"
    elif predicted_value < thresholds[2]:
        severity = "Moderate depression"
    elif predicted_value < thresholds[3]:
        severity = "Moderately severe depression"
    else:
        severity = "Severe depression"

    return predicted_value, severity


# Create Flask application
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chatbot", methods=["POST"])
def chatbot():
    global conversation_history

    data = request.get_json()
    query = data["query"]

    output = chatbot_response(query)
    conversation_history.append({"user": query, "chatbot": output})

    sum, cnt, results = process_query(query, 0, 0, [])
    predicted_value, counter = depression_measure([query], 0, 0)

    label, score = roberta_classifier(str(query))

    response = {
        "response": output,
        "emotion_analysis": results,
        "depression_level": predicted_value / counter if counter > 0 else 0,
        "conversation_history": conversation_history,
        "emotions": [[label, score]],
    }

    if any(
        phrase in query.lower()
        for phrase in [
            "thank you",
            "bye",
            "goodbye",
            "see you later",
            "see you soon",
            "take care",
        ]
    ):
        avg_depression_level, severity = severity_levels(predicted_value, counter)
        overall_emotional_quotient = sum / cnt if cnt > 0 else 0
        response["average_depression_level"] = avg_depression_level
        response["severity"] = severity
        response["overall_emotional_quotient"] = overall_emotional_quotient

    return jsonify(response)


@app.route("/favicon.ico")
def favicon():
    return "", 204


if __name__ == "__main__":
    conversation_history = []
    app.run(debug=True, port=8080, use_reloader=False)
