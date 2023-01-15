import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import re


def cleaning(text):
    text = text.lower()
    text = re.sub(' +', ' ', text)  # removing multiple spaces
    text = re.sub(r'[^\w\s]', '', text)  # removing special characters
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop]
    text = ' '.join(tokens)
    return text




tfidf_vectorizer = TfidfVectorizer(encoding='utf-8',
                                   ngram_range=(1, 2),
                                   stop_words=None,  # We cleaned them already
                                   lowercase=False,  # We converted all to lowercase
                                   max_df=0.95,
                                   min_df=10,
                                   norm='l2',
                                   sublinear_tf=True)

if os.name == 'nt':
    label_encoder_path = 'model\\label_encoder.pkl'
    trained_model_path = 'model\\trained_model.pkl'
    trained_vectorizer_path = 'model\\trained_vectorizer.pkl'

else:
    label_encoder_path = 'model/label_encoder.pkl'
    trained_model_path = 'model/trained_model.pkl'
    trained_vectorizer_path = 'model/trained_vectorizer.pkl'


def load_trained_models():
    label_encoder = pickle.load(open(label_encoder_path, 'rb'))
    trained_model = pickle.load(open(trained_model_path, 'rb'))
    trained_vectorizer = pickle.load(open(trained_vectorizer_path, 'rb'))
    return label_encoder, trained_vectorizer, trained_model


def lets_predict(lr_model, trained_vectorizer, fresh_data):
    fresh_data_df = pd.DataFrame([fresh_data], columns=['Text'])
    fresh_data_df['Text'] = fresh_data_df['Text'].apply(cleaning)
    fresh_data_values = fresh_data_df['Text'].values
    fresh_data_transformed = trained_vectorizer.transform(fresh_data_values)

    return lr_model.predict(fresh_data_transformed.toarray())

# **Trying one more Sample record**

# sample = pd.DataFrame(["Months before the assembly election in Karnataka, Bharatiya Janata Partys (BJP) senior leader and the state former Chief Minister SM Krishna announced his retirement from active politics on Wednesday"],columns=['Text'])
# sample = "Months before the assembly election in Karnataka, Bharatiya Janata Partys (BJP) senior leader and the state former Chief Minister SM Krishna announced his retirement from active politics on Wednesday"
# #
# trained_vectorizer, trained_model = load_trained_models()
# #
# print(lets_predict(trained_model, trained_vectorizer, sample))
