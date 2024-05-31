from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import pickle
import random
# import h5py
# import json
from collections import defaultdict


from transformers import pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

# from collections import defaultdict
from intent import intents
from dotenv import load_dotenv

# To supress
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# The langchain part -->
load_dotenv()
x = os.getenv('HUGGINGFACE_API_TOKEN')
encodings = ['utf-8', 'latin-1', 'utf-16']


def remove_newlines(text):
    """Remove newline characters from the text."""
    return text.replace('\n', '')
def replace_slash_with_or(text):
    """Replace '/' with 'or' in the text."""
    return text.replace('/', ' or ')
def remove_brackets(text):
    """Remove brackets from the text."""
    return text.replace('(', '').replace(')', '')
def replace_dashes_with_space(text):
    """Replace '____' with a space in the text."""
    return text.replace('____', '')
def replace_multiple_dash_with_space_and_respective_text(text):
    # Replace multiple underscores with an empty string
    text = text.replace('____', '')

    # Replace single underscore with "what is your choice.."
    if '_' in text:
        text = text.replace('_', 'what is your choice')

    return text
def preprocess_text(text):
    """Preprocess the text using all defined functions."""
    text = remove_newlines(text)
    text = replace_slash_with_or(text)
    text = remove_brackets(text)
    text = replace_dashes_with_space(text)
    text = replace_multiple_dash_with_space_and_respective_text(text)
    return text

for encoding in encodings:
    try:
        loader = TextLoader(r"VULNERABILITY_TO_DEPRESSION.txt", encoding=encoding)
        depression_document = loader.load()
        print("Document loaded successfully using encoding:", encoding)
        break  # Break out of the loop if successful
    except Exception as e:
        print("Error loading document with encoding:", encoding)
        print(e)

# trying to preprocess text from a document about depression
final_text=preprocess_text(str(depression_document[0]))
text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=1000,
    chunk_overlap=300,
    length_function=len,
    is_separator_regex=False,
)

depression_docs = text_splitter.split_documents(depression_document)

embeddings=HuggingFaceEmbeddings()
db = FAISS.from_documents(depression_docs,embeddings)

# FLAN-T5 is a family of large language models trained at Google, 
huggingface_hub = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.8, "max_length": 2048}, huggingfacehub_api_token=x)

# Load the question-answering chain
chain = load_qa_chain(huggingface_hub, chain_type="stuff")



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

# results_all = []

# Process the query for emotion and depression analysis
def process_query(query, sum, cnt, results):
    label, score = roberta_classifier(str(query))
    quotient = float(score) * 100
    sum += quotient
    results.append({"query": query, "emotion": label, "score": quotient})
    cnt += 1
    return sum, cnt, results


def overall_emotional_quotient(results):
    category_data = defaultdict(lambda: [0, 0])
    for category, value in results:
        category_data[category][0] += value
        category_data[category][1] += 1
    averages = {category: sum_value / count for category, (sum_value, count) in category_data.items()}
    max_category = max(averages, key=averages.get)
    max_average = averages[max_category]
    print("Category with the highest average:", max_category)
    print("Average value:", max_average)
    return max_category, max_average

def emotional_quotient_avg_each_cat(results):
    emotion_data = defaultdict(lambda: [0, 0])
    for emotion, value in results:
        emotion_data[emotion][0] += value
        emotion_data[emotion][1] += 1
    averages = {emotion: sum_value / count for emotion, (sum_value, count) in emotion_data.items()}
    # for emotion, average in averages.items():
    #     print(f"Average for {emotion}: {average}")
    return averages


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
CORS(app)


@app.route("/chatbot", methods=["POST"])
def chatbot():
    global conversation_history
    global results_all

    data = request.get_json()
    query = data["query"]

    output = chatbot_response(query) # Here is the input coming from the intent_chatbot

    docsResult = db.similarity_search(query)
    # print(f"answring the questions through precise manner :{chain.run(input_documents=docsResult,question = query)}")
    print("broad result:")
    print(preprocess_text(str(docsResult[0].page_content)))

    output_l = preprocess_text(str(docsResult[0].page_content))

    conversation_history.append({"user": query, "chatbot": output, "langchain" : output_l})

    sum, cnt, results = process_query(query, 0, 0, [])


    print("results_all\n",results_all, "\n")

    predicted_value, counter = depression_measure([query], 0, 0)

    label, score = roberta_classifier(str(query))

    results_all.append([label, score * 100]) 

    response = {
        "conversation_history": conversation_history,
        "depression_level": predicted_value / counter if counter > 0 else 0, #ok
        "emotion_analysis": {"emotions" : label, "score": (score * 100)}, #ok
        "response_lagchain": output_l, #ok
        "response": output, #ok
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
        overall_emotional_quotient_vlaue = overall_emotional_quotient(results_all)
        emotional_quotient_avg_each_catagory_value = emotional_quotient_avg_each_cat(results_all)
        response["average_depression_level"] = avg_depression_level
        response["severity"] = severity
        response["overall_emotional_quotient"] = overall_emotional_quotient_vlaue
        response["emotional_quotient_avg_each_catagory_value"] = emotional_quotient_avg_each_catagory_value

    print('\n')
    print(response)
    print('\n')

    return jsonify(response)


# @app.route("/favicon.ico")
# def favicon():
    return "", 204


if __name__ == "__main__":
    conversation_history = []
    results_all = []
    app.run(debug=True, port=8080, use_reloader=False)
