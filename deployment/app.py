"""Researchly Research Abstract Classifier - Flask web app (Question 2.3)."""
import os
import pickle

from flask import Flask, render_template, request

from preprocessing import clean

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, 'models', 'best_model.pickle')

app = Flask(__name__)

# Load the best model (TF-IDF -> SMOTE -> Linear SVM pipeline)
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    abstract = request.form.get('abstract', '').strip()
    if not abstract:
        return render_template('index.html', error="Please enter a research abstract first.")

    prediction = model.predict([clean(abstract)])[0]
    label = 'artificial intelligence' if prediction == 1 else 'classical research'
    return render_template('index.html', abstract=abstract, prediction=label)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
