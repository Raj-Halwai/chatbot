#Helper functions for NLP preprocessing using NLTK

import re 
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK resources (only first time)
nltk.download('stopwords')

# Initialize stemmer and stopword list
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocess text: clean, remove stopwords, stem
def preprocess(text):
    text = text.lower()  # lowercase
    text = re.sub(r"\d+", "", text)  # remove digits
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)