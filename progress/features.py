import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


###########################################################
# Helper Functions
def getPostivePath(p):
    return Path("./p-progress/Data/Train/Positive") / p
def getNegativePath(p):
    return Path("./p-progress/Data/Train/Negative") / p
def getFileContent(fp):
    with open(fp) as f:
        content = f.read()
    return content

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
        self.CountVectorizer = CountVectorizer(stop_words="english", analyzer='word', ngram_range=(1, 3), min_df=0.001)
        self.categories = ["Restaurants-food", "yelp-food"]
        self.positive_files = map(getPositivePath, self.categories)
        self.negative_files = map(getNegativePath, self.categories)
        self.positive_frequencies = []
        self.negative_frequencies = []

    def fit(self, X):
        self.tfidfVectorizer.fit([' '.join(X)])
        for fp in self.positive_files:
            content = getFileContent(fp)
            frequencies = npasarray(self.CountVectorizer.fit_transform(content)).sum(axis=0)
            featureNames = self.CountVectorizer.get_feature_names()
            self.positive_frequencies.append(dict(zip(featureNames, frequencies)))
        for fp in self.negative_files:
            content = getFileContent(fp)
            frequencies = npasarray(self.CountVectorizer.fit_transform(content)).sum(axis=0)
            featureNames = self.CountVectorizer.get_feature_names()
            self.negative_frequencies.append(dict(zip(featureNames, frequencies)))
        
    def transform(self, X):
        return self.tfidfVectorizer.transform(X)
############################################################
