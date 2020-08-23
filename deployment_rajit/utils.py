import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from unidecode import unidecode
from numpy.linalg import norm
import numpy as np
import math
import re
from preprocess import pre_process


def compute_average(vec_list):
    """
    Takes a list of vectors and returns their average
    """
    return np.sum(vec_list, axis = 0)/len(vec_list)

#some similarity functions
def compute_euclidean_dist(vec1, vec2):
    """Computes the Euclidean similarity between two vectors"""
    assert len(vec1) == len(vec2)
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.sqrt(np.sum(np.square(vec2 - vec1)))

def compute_cosine_sim(vec1, vec2):
    """Computes the cosine similarity between vec1, vec2"""

    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2)/(norm(vec1) * norm(vec2))

def preproc_user_input(txt, model):
    """
    Applies the preprocessing steps to user input.
    
    Args:
        txt (string): a string representing user input
        model (Gensim trained model object): a trained Gensim Word2vec model
    
    Returns:
        string: a preprocessed user input string
        
    preproc_user_input applies the same preprocessing that was applied to the catalog. 
    However, it also applies additional preprocessing by removing words that are not
    in the model vocabulary.
    
    """
    txt = pre_process(txt)
    txt_tokenized =  [word for word in txt.split(" ") if word in model.wv.vocab]
    return " ".join(txt_tokenized)

def compute_user_input_embedding(txt, model):
    """
    Computes a sentence2vece embedding for preprocessed user input.
    
    Args:
        txt (string): preprocessed user input (all words are assumed to be in the model vocab) 
    
    Returns:
        list: a Python list representing the embedding for txt        
    """
    embeddings = []
    tokens = txt.split(" ")
    for word in tokens:
        embeddings.append(model.wv[word])
    sentence_embedding = compute_average(embeddings)
    return sentence_embedding

def get_similar_products(user_input_emb, ref_catalog, n = 5):
    """
    Returns the n most similar products for a given user input embedding.
    
    Args:
        user_input_emb (list): Represents the user input embedding in the model vector space
        n (int): the number of top n recommendations desired
        ref_catalog (Pandas dataframe): A pandas dataframe containing product descriptions
    
    Returns:
        Python list: a list  of n elements 
        containing tuples of the following form (product_id, cosine_similarity)
    """
    sim_list = []
    for i in range(len(ref_catalog)):
        desc_id = ref_catalog.iloc[i]['id']
        emb = ref_catalog.iloc[i]['desc_embedding']
        cos_sim = compute_cosine_sim(emb,user_input_emb)
        sim_list.append((desc_id, cos_sim))
    top_n = sorted(sim_list, key= lambda tup: tup[1], reverse = True)[:n]
    return top_n


def serve_recos(ids, ref_catalog):
    """
    Provides recommendations in natural text format based on raw descriptions in the ecomm dataset.
    
    Args:
        ids (list of ints): Contains ids whose descriptions are to be fetched
        ref_catalog (Pandas dataframe): a Pandas dataframe containing product descriptions
    
    Returns:
        List: this list contains the descriptions from ecomm dataset for ids provided in argument 
    """
    desc_list = []
    for desc_id in ids:
        desc_list.append(ref_catalog[ref_catalog['id'] == desc_id].iloc[0]['description'])
    return desc_list

def remove_html(txt):
    """
    Removes html tags from txt
    
    Args:
        txt (string): a string from which html tags are to be removed
    
    Returns:
        string: a cleaned string with no html tags
    """
    TAG_RE = re.compile(r'<[^>]+>')
    return TAG_RE.sub("", txt).strip()

if __name__ == "__main__":
    vec1 = [1,2,3]
    vec2 = [4,5,6]

    print(compute_cosine_sim(vec1, vec2))