# Im including this file to keep note of the Sequence Mathching Parser
import pandas as pd
import nltk
import re
import numpy as np
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import  WordNetLemmatizer
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem.porter import PorterStemmer
import string
import re
from sklearn.feature_extraction import text
from preprocess import *
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
from difflib import SequenceMatcher
import pickle as pkl

x_list = [w for w in df_train['description_x']]
y_list = [w for w in df_train['description_y']]


vectorizer = text.TfidfVectorizer()

def cosine_sim(test1, test2):
    tfidf = vectorizer.fit_transform([test1, test2])
    result = ((tfidf * tfidf.T).A)[0,1]
    return result
    
def cosine_sim_df(df_data):
    col1 = 'description_x'
    col2 = 'description_y'
    df_data[col1] = df_data[col1].str.replace(r'\d', '')
    df_data[col2] = df_data[col2].str.replace(r'\d', '') 
    df_data['cos_sim'] = 0
    df_data['cos_sim'] = df_data.apply(
        lambda x: cosine_sim(x[col1], x[col2]), axis=1)
    return df_data

thresholds = [0.4,0.6,0.8]
df_result = cosine_sim_df(df_train)

def seq_match(test1, test2):
    return SequenceMatcher(None, test1, test2).ratio()
    
def seq_match_df(df_data):
    col1 = 'description_x'
    col2 = 'description_y'
    df_data[col1] = df_data[col1].str.replace(r'\d', '')
    df_data[col2] = df_data[col2].str.replace(r'\d', '')
    
    df_data['seq_match'] = 0
    df_data['seq_match'] = df_data.apply(
        lambda x: seq_match(x[col1], x[col2]), axis=1)
    
    return df_data

df_result = seq_match_df(df_test)

df_result['same_security'] = df_result['seq_match'] > 0.5    
df_group = df_result.groupby('same_security').size().reset_index()
df_group.columns = ['correct', 'cnt']

df_result = seq_match_df(df_train)

### Alternative Model ###
import gensim
f = x_list
  
data = [] 
  
# iterate through each sentence in the file 
for k in f:    
    for i in sent_tokenize(k): 
        temp = [] 

        # tokenize the sentence into words 
        for j in word_tokenize(i): 
            temp.append(j.lower()) 

        data.append(temp) 
  
# Create CBOW model 
model1 = gensim.models.Word2Vec(data, min_count = 1,  
                              size = 100, window = 5) 
model1.train([corpus], total_examples= model1.corpus_count, epochs=10)
  
# Create Skip Gram model 
model2 = gensim.models.Word2Vec(data, min_count = 1, size = 100, 
                                             window = 5, sg = 1) 

filename_1 = 'model1.pkl'
pkl.dump(model1, open('model1.pkl', 'wb'))
