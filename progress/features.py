import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

############################################################
# Abstract base class
class FeatureExtractor:
    def fit(self, X):
        raise NotImplementedError("Override me")
    def transform(self, X):
        raise NotImplementedError("Override me")
############################################################

# Basic feature extractor that uses scikit-learn's TfidfVectorizer
# to create ngram features with tf-idf scores as feature values
class FeatureExtractorV0(FeatureExtractor):
    def __init__(self):
        self.tfidfVectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=0.001, stop_words='english')
    
    def fit(self, X):
        self.tfidfVectorizer.fit([' '.join(X)])
    
    def transform(self, X):
        return self.tfidfVectorizer.transform(X)
############################################################

############################################################

# Version-1 Feature extractor that uses scikit-learn's TfidfVectorizer
# to create ngram features with tf-idf scores as feature values and 
# custom implementation of PMI scores using scikit-learn's CountVectorizer
# To evaluate the n-gram frequencies. 
class FeatureExtractorV1(FeatureExtractor):
    def __init__(self):
        self.tfidfVectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=0.001, stop_words='english')

    def fit(self, X):
        self.tfidfVectorizer.fit([' '.join(X)])

    def transform(self, X):
        return self.tfidfVectorizer.transform(X)
############################################################
