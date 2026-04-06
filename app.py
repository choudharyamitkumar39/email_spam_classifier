from flask import Flask, render_template, request
import pickle
import nltk
import string
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize
app = Flask(__name__)
ps = PorterStemmer()

# Load model and vectorizer (safe path for deployment)
model = pickle.load(open(os.path.join(os.path.dirname(__file__), 'model.pkl'), 'rb'))
tfidf = pickle.load(open(os.path.join(os.path.dirname(__file__), 'vectorizer.pkl'), 'rb'))

# Download NLTK data (needed for Render)
nltk.download('punkt')
nltk.download('stopwords')

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Home route
@app.route('/')
def home():
    return render_template('index.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    input_sms = request.form.get('message')

    # ✅ Handle empty input
    if not input_sms or input_sms.strip() == "":
        return render_template('index.html', prediction=None)

    # 1. preprocess
    transformed_sms = transform_text(input_sms)

    # 2. vectorize (FIXED sparse → dense issue)
    vector_input = tfidf.transform([transformed_sms]).toarray()

    # 3. predict
    result = model.predict(vector_input)[0]

    if result == 1:
        prediction = "Spam"
    else:
        prediction = "Not Spam"

    return render_template('index.html', prediction=prediction)


# Run locally
if __name__ == '__main__':
    app.run(debug=True)