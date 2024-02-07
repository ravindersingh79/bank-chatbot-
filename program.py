from flask import Flask, render_template, request, jsonify
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

app = Flask(__name__)

data_file = open('intent.json').read()
intents = json.loads(data_file)

lemmatizer = WordNetLemmatizer()

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model2.h5')


def cleaning(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = cleaning(sentence)
    bag = [0] * len(words)
    for i, word in enumerate(words):
        if word in sentence_words:
            bag[i] = 1
    return np.array(bag)


def predict_class(sentence, error_threshold=0.25):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/sendMessage', methods=['POST'])
def send_message():
    if request.method == 'POST':
        user_message = request.form['message']

        intents_list = predict_class(user_message)
        response = get_response(intents_list, intents)

        return jsonify({'user': user_message, 'bot': response})


if __name__ == '__main__':
    app.run(debug=True)
