import pandas as pd
import nltk
nltk.download('punkt')
import re
import numpy as np
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem.porter import PorterStemmer
import string
import re

# list of stop words and punctuation in the English language
STOPWORDS = stopwords.words('english') + list(string.punctuation)
lemmatizer = WordNetLemmatizer()

def pre_process(txt):
    """
    Preprocesses a string by lowering case, removing stopwrds, removing punctuation, lemmatizing
    
    Args:
        txt: a string to be preprocessed
    
    Returns:
        preprocessed string     
    """
    TAG_RE = re.compile(r'<[^>]+>')
    NUM_RE = re.compile(r'[0-9]+')
    txt = TAG_RE.sub("", txt)
    txt = NUM_RE.sub("", txt)
    txt = txt.replace('``',"")
    txt = txt.lower()
    txt = unidecode(txt)
    txt_tokenized = word_tokenize(txt)
    txt_tokenized = [item for item in txt_tokenized if item not in STOPWORDS]
    for index, word in enumerate(txt_tokenized):
        txt_tokenized[index] = lemmatizer.lemmatize(word)
    return " ".join(txt_tokenized)


if __name__ == "__main__":
    #import dataset
    ecomm = pd.read_csv('Data/sample-data.csv')
    #preprocess descriptions
    ecomm['preproc_description'] = ecomm['description'].apply(pre_process)
    #write back to the data/
    ecomm.to_csv('Data/ecomm_preprocessed.csv')
