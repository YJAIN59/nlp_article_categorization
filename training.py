# %%
import os
import pandas as pd
import matplotlib.pyplot as plt


# # %%
# data_path = 'D:\\ML\\article_sort\\learn-ai-bbc\\BBC News Train.csv'
#
# # %%
# data_raw = pd.read_csv(data_path)
# #data_raw = data_raw.loc[np.random.choice(data_raw.index, size=2000)]
# data_raw.shape
#
# # %%
# print("Number of rows in data =",data_raw.shape[0])
# print("Number of columns in data =",data_raw.shape[1])
# print("\n")
# print("**Sample data:**")
# data_raw.head()
#
# # %% [markdown]
# # Check for null values:
#
# # %%
# data_raw.isnull().sum()
#
# # %% [markdown]
# # Checking number of articles in each category:
#
# # %%
# # categories_list = data_raw['Category'].unique()
# print(data_raw['Category'].value_counts())
# print(data_raw['Category'].value_counts(normalize=True))
#
# # %% [markdown]
# # **Normal Data Exploration**
#
# # %%
# data_raw['Category'].value_counts().plot(kind='bar')
#
# # %%
# data_raw['Text'].where(data_raw['ArticleId'].duplicated()==False)
#
# # %%
# data_raw[data_raw[['Text', 'Category']].duplicated()==True]
#
# # %%
# data_raw = data_raw.drop_duplicates(subset=['Text', 'Category'])
#
# # %%
# data_raw.shape

# %% [markdown]
# **NLP Data Exploration**

# %%
# import nltk
# nltk.download('stopwords', download_dir=os.curdir)
from nltk.corpus import stopwords
stop=set(stopwords.words('english'))

# # %%
# from wordcloud import WordCloud
#
# # %%
# comment_words = ''
# for val in data_raw['Text'][data_raw['Category']=='business']:
#
#     # typecaste each val to string
#     val = str(val)
#
#     # split the value
#     tokens = val.split()
#
#     # Converts each token into lowercase
#     for i in range(len(tokens)):
#         tokens[i] = tokens[i].lower()
#
#     comment_words += " ".join(tokens)+" "
#
# # %%
# wordcloud = WordCloud(
#         background_color='black',
#         stopwords=stop,
#         max_words=30,
#         max_font_size=100,
#         scale=5,
#         random_state=1)
#
# wordcloud=wordcloud.generate(str(comment_words))
#
# # %%
# plt.imshow(wordcloud)

# %% [markdown]
# **Preprocessing**

# %%
import nltk
from nltk.tokenize import word_tokenize
# nltk.download('punkt', download_dir=os.curdir)
import re

# %%
# word_tokenize(' i have had a  yes  from a couple of others should i need them. davies impressed during. ')
#
# # %%
# print(re.sub(' +', ' ', ' i have had a  yes     from a couple'))
# print(re.sub(r'[^\w\s]', '', ' i %^ & have had a  yes     from a couple'))
# print(re.sub(r'[^a-zA-Z0-9\s]', '', ' I AM i %^ & have had a  yes     from a couple'))


# %%
def cleaning(text):
    text = text.lower()
    text = re.sub(' +', ' ', text) # removing multiple spaces
    text = re.sub(r'[^\w\s]', '', text) # removing special characters
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop]
    text = ' '.join(tokens)
    return text

# %%
# cleaning('i have had a  yes  from a couple of others should i need them. davies impressed during. ')
#
# # %%
# data_raw['Text'] = data_raw['Text'].apply(cleaning)
#
# # %%
# data_raw['Text'].head()

# %% [markdown]
# **Analysing tf-idf**

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# %% [markdown]
# **Preprocessing**
#
# # %%
# from sklearn.preprocessing import LabelEncoder
# lbl_encoder = LabelEncoder()
# data_raw['Category_nm'] = lbl_encoder.fit_transform(data_raw['Category'])
#
# # %%
# data_raw[['Category','Category_nm']].value_counts()

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from time import time
from sklearn.metrics import accuracy_score

# %%
tfidf_vectorizer = TfidfVectorizer(encoding = 'utf-8',
                             ngram_range = (1,2),
                             stop_words= None,          # We cleaned them already
                             lowercase = False,         # We converted all to lowercase
                             max_df = 0.95,
                             min_df = 10,
                             norm = 'l2',
                             sublinear_tf = True)


# %%
# def logistic_regression(test_size=0.1):
#     X_train, X_test, y_train, y_test = train_test_split(data_raw['Text'], data_raw['Category_nm'], test_size=test_size, random_state=42)
#     t0 = time()
#     # tfidf_vectorizer.fit(X_train)
#     X_train = tfidf_vectorizer.fit_transform(X_train)
#     y_train = lbl_encoder.fit_transform(y_train)
#     X_test = tfidf_vectorizer.transform(X_test)
#     y_test = lbl_encoder.transform(y_test)
#     logistic_regression = LogisticRegression(C=1e5, multi_class='multinomial')
#     lr_model = logistic_regression.fit(X_train, y_train)
#
#     acc = accuracy_score(y_test, lr_model.predict(X_test))
#     pred = lr_model.predict(X_test)
#     print("done in %0.3fs." % (time() - t0))
#     return acc, pred, y_test, lr_model

# # %%
# X_train, X_test, y_train, y_test = train_test_split(data_raw['Text'], data_raw['Category_nm'], test_size=0.2, random_state=42)
#
# # %%
# lr_df = pd.DataFrame(columns = ['Test Size','Accuracy', 'Predictions', 'Test'])
# for i in np.arange(0.1,1.0,0.1):
#     acc, pred, test, lr_model = logistic_regression(i)
#     lr_df.loc[len(lr_df.index)] = [i*100, acc, pred, test]
# lr_df
#
# # %%
# acc, pred, test, lr_model = logistic_regression(0.2)

# **Saving Model**
# def saving_trained_model():
#     pickle.dump(lr_model, open('D:\ML\\article_sort\model\\trained_model.pkl','wb'))

# **Load Model**

# pickle.dump(tfidf_vectorizer, open('D:\ML\\article_sort\model\\trained_vectorizer.pkl','wb'))

def load_trained_models():
    label_encoder = pickle.load(open('D:\ML\\article_sort\model\\label_encoder.pkl','rb'))
    trained_model = pickle.load(open('D:\ML\\article_sort\model\\trained_model.pkl','rb'))
    trained_vectorizer = pickle.load(open('D:\ML\\article_sort\model\\trained_vectorizer.pkl','rb'))
    return label_encoder, trained_vectorizer, trained_model
# **Testing Model**

# %%
# data_path_2 = 'D:\\ML\\article_sort\\learn-ai-bbc\\BBC News Test.csv'
# data_raw_2 = pd.read_csv(data_path_2)
# #data_raw = data_raw.loc[np.random.choice(data_raw.index, size=2000)]
# data_raw_2.shape

# %%
def lets_predict(lr_model, trained_vectorizer, fresh_data):
    fresh_data_df = pd.DataFrame([fresh_data], columns=['Text'])
    fresh_data_df['Text'] = fresh_data_df['Text'].apply(cleaning)
    fresh_data_values = fresh_data_df['Text'].values
    fresh_data_transformed = trained_vectorizer.transform(fresh_data_values)
    
    # return trained_model.predict(fresh_data_df)
    return lr_model.predict(fresh_data_transformed.toarray())
    

# %%
# lets_predict(load_trained_model(), data_raw_2['Text'])

# %% [markdown]
# **Trying one more Sample record**

# sample = pd.DataFrame(["Months before the assembly election in Karnataka, Bharatiya Janata Partys (BJP) senior leader and the state former Chief Minister SM Krishna announced his retirement from active politics on Wednesday"],columns=['Text'])
# sample = "Months before the assembly election in Karnataka, Bharatiya Janata Partys (BJP) senior leader and the state former Chief Minister SM Krishna announced his retirement from active politics on Wednesday"
# #
# trained_vectorizer, trained_model = load_trained_models()
# #
# print(lets_predict(trained_model, trained_vectorizer, sample))


