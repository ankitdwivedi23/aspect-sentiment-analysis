import numpy as np
import pandas as pd
import math
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csc_matrix

###########################################################
# Helper Functions

def getPath(p, context):
    if context == "pos":
        return getPositivePath(p)
    else:
        return getNegativePath(p)
    
def getPostivePath(p):
    return Path("./p-progress/Data/Train/Positive") / p

def getNegativePath(p):
    return Path("./p-progress/Data/Train/Negative") / p

def getFileContent(fp):
    with open(fp) as f:
        content = f.read()
    return content

def negateContext(c):
    if c == "pos":
        return "neg"
    else
        return "pos"

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
        self.positive_frequencies = {}
        self.negative_frequencies = {}
        self.all_frequencies = {}
        self.N_pos = {}
        self.N_neg = {}
        self.N = {}

    def fit(self, X):
        self.tfidfVectorizer.fit([' '.join(X)])
        
        def getFrequencyDictionaryAndN(filePath):
            content = getFileContent(filePath)
            frequencies = npasarray(self.CountVectorizer.fit_transform(content)).sum(axis=0)
            featureNames = self.CountVectorizer.get_feature_names()
            return dict(zip(featureNames, frequencies)), len(self.CounVectorizer.vocabulary.keys())

        for c in self.categories:
            self.positive_frequencies[c], self.N_pos[c] = getFrequencyDictionaryAndN(getPath(c, "pos"))
            self.negative_frequencies[c], self.N_neg[c] = getFrequencyDictionaryAndN(getPath(c, "neg"))
            self.all_frequencies[c] = { k: x.get(k, 0) + y.get(k, 0) for k in set(self.positive.frequencies[c]) | set(self.negative.frequencies[c]) }
            self.N = self.N_pos[c] + self.N_neg[c]
    
    def getFrequencyDictionaryAndN(context):
        if context == "pos":
            return self.positive_frequencies, self.N_pos
        elif context == "neg":
            return self.negative_frequencies, self.N_neg
        return self.all_frequencies, self.N

    def pmi_score(w, context):
        pmi_score = {}
        frequency_context, N_context = getFrequencyDictionaryAndN(context)
        frequency_all, N = getFrequencyDictionaryAndN("all")
        for c in self.categories:
            pmi_score[c] = math.log2( (N[c]*frequency_context[c][w])/(N_context[c]*frequency_all[c]) )
        return pmi_score
    
    def get_pmi_based_score(w):
        pmi_score_pos = pmi_score(w, "pos")
        pmi_score_neg = pmi_score(w, "neg")
        return { k: x.get(k, 0) - y.get(k, 0) for k in set(pmi_score_pos) & set(pmi_score_neg) }

    def transform(self, X):
        feature_names = self.tfidfVectorizer.get_feature_names()
        tfdif_matrix = self.tfidfVectorizer.transform(X).toarray()
        matrix = []
        for i in range(len(X)):
            matrix[i] = tdif_matrix[i]
            for f_n, feature in enumerate(feature_names):
                pmi_scores = get_pmi_based_score(feature)
                for c in self.categories:
                    matrix[i].append(pmi_scores[c])
        return csc_matrix(matrix)

############################################################
