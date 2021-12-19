# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 15:02:27 2021

@author: SAMARTH P SHET
"""

" Text Sentiment Analysis Using Amazon Review Data"

# importing the libraries related to the data manipulation.
import numpy as np
import pandas as pd
import string

# importing the libraries related to the data_visualization.
import seaborn as sns
import matplotlib.pyplot as plt

# TextBlob is a Python library for processing textual data.
from textblob import TextBlob

# To avoid warnings
import warnings
warnings.filterwarnings("ignore")

# datapath = E:\DOWNLOADS\book_review.csv
Book_review = pd.read_csv(r'E:\DOWNLOADS\book_review.csv')
Book_review

# shape of the actual dataframe
Book_review.shape

# display the names of various columns present in the dataframe
Book_review.columns

# Find the information about the given dataFrame including the index dtype and column dtypes, non-null values and memory usage.
Book_review.info()

# Finding the total number of null values, if present
Book_review.isnull().sum()

"Considering only the review_text feature of the dataframe for our further analysis:"
pd.options.mode.chained_assignment = None
df = Book_review[["review_text"]]
df["review_text"] = df["review_text"].astype(str)
df.head()

# shape of the review data
df.shape

df["lowercase_text"] = df["review_text"].str.lower()

Punctuation_remove = string.punctuation
def remove_punctuation(lowercase_text):
    """custom function to remove the punctuation"""
    return lowercase_text.translate(str.maketrans('', '', Punctuation_remove))

df["NoPunctuations_text"] = df["lowercase_text"].apply(lambda lowercase_text: remove_punctuation(lowercase_text))

# importing the NLP library
import nltk
from nltk.corpus import stopwords
stopwords

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
print(stopwords)

def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in stopwords])

df["Nostopwords_text"] = df["NoPunctuations_text"].apply(lambda text: remove_stopwords(text))

from collections import Counter
cnt = Counter()
for text in df["Nostopwords_text"].values:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(10)

FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
def remove_freqwords(text):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])

df["NoFreqwords_text"] = df["Nostopwords_text"].apply(lambda text: remove_freqwords(text))

n_rare_words = 10
RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])
def remove_rarewords(NoFreqwords_text):
    """custom function to remove the rare words"""
    return " ".join([word for word in str(NoFreqwords_text).split() if word not in RAREWORDS])

df["NoRareFreqwords_text"] = df["NoFreqwords_text"].apply(lambda NoFreqwords_text: remove_rarewords(NoFreqwords_text))

from nltk.stem.porter import PorterStemmer

# Drop the four columns 
df.drop(["NoFreqwords_text", "lowercase_text", "NoPunctuations_text","Nostopwords_text"], axis=1, inplace=True) 

stemmer = PorterStemmer()
def stem_words(NoRareFreqwords_text):
    return " ".join([stemmer.stem(word) for word in NoRareFreqwords_text.split()])

df["Stemmed_text"] = df["NoRareFreqwords_text"].apply(lambda NoRareFreqwords_text: stem_words(NoRareFreqwords_text))

nltk.download('wordnet')
  
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
def lemmatize_words(Stemmed_text):
    return " ".join([lemmatizer.lemmatize(word) for word in Stemmed_text.split()])

df["Lemmatized_text"] = df["Stemmed_text"].apply(lambda Stemmed_text: lemmatize_words(Stemmed_text))


# importing regular expression
import re
def remove_urls(Lemmatized_text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', Lemmatized_text)

df["URL_removed_text"] = df["Lemmatized_text"].apply(lambda Lemmatized_text: remove_urls(Lemmatized_text))

def remove_html(URL_removed_text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', URL_removed_text)

df["Tags_Removed_text"] = df["URL_removed_text"].apply(lambda URL_removed_text: remove_html(URL_removed_text))
df.head()

df.drop(["NoRareFreqwords_text", "Stemmed_text", "Lemmatized_text", "URL_removed_text"], axis=1, inplace=True)
df.head()

from textblob import TextBlob

def getSubjectivity(Tags_Removed_text):
    return TextBlob(Tags_Removed_text).sentiment.subjectivity
    
def getPolarity(Tags_Removed_text):
    return TextBlob(Tags_Removed_text).sentiment.polarity

df ['polarity'] = df['Tags_Removed_text'].apply(getPolarity)
df['subjectivity'] = df['Tags_Removed_text'].apply(getSubjectivity)

def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'
    
df['Analysis_labels'] = df['polarity'].apply(lambda x: getAnalysis(x))
        
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(df['Tags_Removed_text'],
                                                 df['Analysis_labels'],
                                                 test_size = 0.2,random_state = 324)

X_train.shape

X_test.shape


df['Analysis_labels'].value_counts()

# importing the library
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 

vect1 = CountVectorizer()
cv_train = vect1.fit_transform(X_train)
cv_test = vect1.transform(X_test)

print(vect1.vocabulary_)

vect2 = TfidfVectorizer()
TF_train = vect2.fit_transform(X_train)
TF_test = vect2.transform(X_test)

TF_train.shape

#import KNN classifer and fit on the Training dataset
from sklearn.neighbors import KNeighborsClassifier
model1 = KNeighborsClassifier()
model1.fit(cv_train,y_train)

# Accuracy score on training dataset
model1.score(cv_train,y_train)

# Accuracy on Test dataset
model1.score(cv_test,y_test)

# Performing prediction on Test dataset
expected = y_test
predicted = model1.predict(cv_test)

# Calculating F1 score
from sklearn import metrics
from sklearn.metrics import f1_score
f1_score(expected, predicted, average='macro')

model2 = KNeighborsClassifier()
model2.fit(TF_train,y_train)

# Accuracy score on training dataset
model2.score(TF_train,y_train)

# Accuracy on Test dataset
model2.score(TF_test,y_test)

# Performing prediction on Test dataset
expected = y_test
predicted = model2.predict(TF_test)

# Calculating F1 score
from sklearn import metrics
from sklearn.metrics import f1_score
f1_scores = f1_score(expected, predicted, average='macro')

print(f1_scores)


import pickle
pickle.dump(model2, open('KNNmodel.pkl', 'wb'))
















