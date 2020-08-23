import pandas as pd
import nltk
import re
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from numpy.linalg import norm
from nltk.stem import  WordNetLemmatizer
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.tokenize import word_tokenize,sent_tokenize
import string
import re
from sklearn.feature_extraction import text
from preprocess import *
from gensim.models import Word2Vec
import pickle as pkl

def compute_average(vec_list):
    """
    Takes a list of vectors and returns their average
    """
    return list(np.sum(vec_list, axis = 0)/len(vec_list))

if __name__ == "__main__":
    ecomm = pd.read_csv('Data/ecomm_preprocessed.csv')
    corpus = [desc.split(" ") for desc in ecomm['preproc_description']]

    model = Word2Vec(corpus, sg = 1, min_count= 1)

    #compute sentence2vec embeddings for descriptions in ecomm
    desc_embeddings = []
    for row in ecomm['preproc_description']:
        embeddings = []
        tokens = row.split(" ")
        for word in tokens:
            embeddings.append(model.wv[word])
        sentence_embedding = compute_average(embeddings)
        desc_embeddings.append(sentence_embedding)

    #add sentence embeddings to ecomm df
    ecomm['desc_embedding'] = desc_embeddings
    ecomm.to_csv('Data/ecomm_embeddings.csv')
    #write model out to models using Pickle
    pkl.dump(model, open('models/w2v_model.pkl', 'wb'))






