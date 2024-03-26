from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd

app = Flask(__name__)
CORS(app)


data = pd.read_csv('dataset.csv')
texts = data['Text']
labels = data['language']


train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)


pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),  
    ('classifier', MultinomialNB())     
])


pipeline.fit(train_texts, train_labels)


@app.route('/predict', methods=['POST'])
def predict_language():
    data = request.json['text']
    predicted_language = pipeline.predict([data])[0]
    return jsonify({'predicted_language': predicted_language})

if __name__ == '__main__':
    app.run(debug=True)
