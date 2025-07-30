from flask import Flask, render_template, request, jsonify, send_file
import pickle
import pandas as pd
import io
import nltk
import string
import os
import logging
import traceback
from langdetect import detect
from deep_translator import GoogleTranslator
from nltk import pos_tag, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# -------------------- Setup --------------------

# Logging
logging.basicConfig(level=logging.DEBUG)

# Flask app
app = Flask(__name__)

# Ensure NLTK data is available
nltk_data_path = os.path.expanduser('~/nltk_data')
nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)

# Load model/vectorizer
MODEL_PATH = 'models/spam_classifier_model.pkl'
VECTORIZER_PATH = 'models/vectorizer.pkl'

try:
    model = pickle.load(open(MODEL_PATH, 'rb'))
    vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
    logging.info("✅ Model and vectorizer loaded.")
except Exception as e:
    logging.error(f"❌ Error loading model/vectorizer: {e}")
    model, vectorizer = None, None

# -------------------- Routes --------------------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        message = request.form['message']
        lang = detect(message)

        # Translate to English if not in English
        if lang != 'en':
            translated_msg = GoogleTranslator(source=lang, target='en').translate(message)
        else:
            translated_msg = message

        length = len(message)
        word_count = len(message.split())
        punct_count = sum(1 for c in message if c in string.punctuation)
        pos_tags_raw = pos_tag(word_tokenize(translated_msg))
        pos_tags = dict(pd.Series([tag for word, tag in pos_tags_raw]).value_counts())

        transformed = vectorizer.transform([translated_msg])

        # Get predicted label directly (assuming your model outputs 'Spam' or 'Ham')
        label = model.predict(transformed)[0]

        # Get confidence for predicted label
        probs = model.predict_proba(transformed)[0]
        # Find index of predicted label in classes
        class_index = list(model.classes_).index(label)
        confidence = round(float(probs[class_index] * 100), 2)

        pos_tags = {str(k): int(v) for k, v in pos_tags.items()}

        return jsonify(
            result=label,
            confidence=confidence,
            lang=lang,
            length=length,
            word_count=word_count,
            punct_count=punct_count,
            pos=pos_tags
        )
    except Exception as e:
        app.logger.error("Error in /predict route: %s", e, exc_info=True)
        return jsonify({'error': str(e)}), 500



@app.route('/batch', methods=['GET', 'POST'])
def batch():
    if model is None or vectorizer is None:
        return "Model or vectorizer not loaded", 500

    if request.method == 'POST':
        try:
            file = request.files.get('file')
            if not file:
                return "No file uploaded", 400

            df = pd.read_csv(file)
            if 'message' not in df.columns:
                return "CSV must contain 'message' column", 400

            df['prediction'] = model.predict(vectorizer.transform(df['message']))
            df['label'] = df['prediction'].map({0: 'Ham', 1: 'Spam'})

            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)

            return send_file(
                io.BytesIO(output.getvalue().encode()),
                download_name="batch_predictions.csv",
                as_attachment=True
            )
        except Exception as e:
            logging.error("❌ Error in /batch:\n" + traceback.format_exc())
            return str(e), 500

    return render_template('batch.html')


@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        try:
            file = request.files.get('file')
            if not file:
                return "No file uploaded", 400

            df = pd.read_csv(file)
            if 'message' not in df.columns or 'label' not in df.columns:
                return "CSV must have 'message' and 'label' columns", 400

            # Retrain
            X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2)
            new_vectorizer = CountVectorizer()
            X_train_vec = new_vectorizer.fit_transform(X_train)

            new_model = MultinomialNB()
            new_model.fit(X_train_vec, y_train)

            # Save updated model
            os.makedirs('models', exist_ok=True)
            pickle.dump(new_model, open(MODEL_PATH, 'wb'))
            pickle.dump(new_vectorizer, open(VECTORIZER_PATH, 'wb'))

            return "✅ Model retrained successfully!"
        except Exception as e:
            logging.error("❌ Error in /train:\n" + traceback.format_exc())
            return str(e), 500

    return render_template('train.html')


# -------------------- Run App --------------------

if __name__ == '__main__':
    app.run(debug=True)
