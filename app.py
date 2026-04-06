from flask import Flask, render_template, request
import pickle
import nltk
import string
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)
ps = PorterStemmer()

# -----------------------------
# NLTK setup (Render-safe)
# -----------------------------
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(nltk_data_path)

nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

# -----------------------------
# Load model + vectorizer
# -----------------------------
model = pickle.load(open(os.path.join(os.path.dirname(__file__), 'model.pkl'), 'rb'))
tfidf = pickle.load(open(os.path.join(os.path.dirname(__file__), 'vectorizer.pkl'), 'rb'))

# -----------------------------
# Text preprocessing
# -----------------------------
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

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # ✅ If someone opens /predict directly → no crash
    if request.method == 'GET':
        return render_template('index.html')

    input_sms = request.form.get('message')

    # ✅ Handle empty input
    if not input_sms or input_sms.strip() == "":
        return render_template('index.html', prediction=None)

    # Preprocess
    transformed_sms = transform_text(input_sms)

    # Convert sparse → dense (VERY IMPORTANT)
    vector_input = tfidf.transform([transformed_sms]).toarray()

    # Predict
    result = model.predict(vector_input)[0]

    prediction = "Spam ❌" if result == 1 else "Not Spam ✅"

    return render_template('index.html', prediction=prediction)


# -----------------------------
# Run locally
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)