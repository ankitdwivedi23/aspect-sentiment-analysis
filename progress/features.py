import numpy as np
import pandas as pd
import math
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csc_matrix
from scipy.sparse import dok_matrix

###########################################################
# Helper Functions

def getPositivePath(p):
    return Path("./../../p-progress/Data/Train/Positive") / (p)

def getNegativePath(p):
        return Path("./../../p-progress/Data/Train/Negative") / (p)

def getPath(p, context):
    if context == "pos":
        return getPositivePath(p)
    else:
        return getNegativePath(p)
    
def getFileContent(fp):
    with open(fp) as f:
        content = f.readlines()
    return content

def negateContext(c):
    if c == "pos":
        return "neg"
    else:
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
        self.categories = ["yelp_ambience", "yelp_food", "yelp_misc", "yelp_price", "yelp_service"]
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
            frequencies = self.CountVectorizer.fit_transform(content).sum(axis=0).tolist()[0]
            featureNames = self.CountVectorizer.get_feature_names()
            d = dict(zip(featureNames, frequencies)), len(featureNames)
            return d

        for c in self.categories:
            self.positive_frequencies[c], self.N_pos[c] = getFrequencyDictionaryAndN(getPath(c, "pos"))
            self.negative_frequencies[c], self.N_neg[c] = getFrequencyDictionaryAndN(getPath(c, "neg"))
            self.all_frequencies[c] = { k: self.positive_frequencies[c].get(k, 0) + self.negative_frequencies[c].get(k, 0) \
                    for k in set(self.positive_frequencies[c]) | set(self.negative_frequencies[c]) }
            self.N[c] = self.N_pos[c] + self.N_neg[c]
        
    def getFrequencyDictionaryAndN(self, context):
        if context == "pos":
            return self.positive_frequencies, self.N_pos
        elif context == "neg":
            return self.negative_frequencies, self.N_neg
        return self.all_frequencies, self.N

    def pmi_score(self, w, context):
        pmi_score = {}
        frequency_context, N_context = self.getFrequencyDictionaryAndN(context)
        frequency_all, N = self.getFrequencyDictionaryAndN("all")
        for c in self.categories:
            if w in frequency_context[c] and w in frequency_all[c]:
                pmi_score[c] = math.log2( (N[c]*frequency_context[c][w])/(N_context[c]*frequency_all[c][w]) )
        return pmi_score
    
    def get_pmi_based_score(self, w):
        pmi_score_pos = self.pmi_score(w, "pos")
        pmi_score_neg = self.pmi_score(w, "neg")
        score = {}
        for c in self.categories:
            score[c] = 0.0
            if c in pmi_score_pos and c in pmi_score_neg:
                score[c] = (pmi_score_pos[c] - pmi_score_neg[c])
        return score

    def transform(self, X):
        feature_names = self.tfidfVectorizer.get_feature_names()
        tfdif_matrix = self.tfidfVectorizer.transform(X).toarray()
        m,n = tfdif_matrix.shape
        l,f = len(self.categories), len(feature_names)
        matrix = dok_matrix((m,(n+(f*l))))
        for i in range(len(X)):
            for k,v in enumerate(tfdif_matrix[i]):
                matrix[i,k] = v
            for f_n, feature in enumerate(feature_names):
                pmi_scores = self.get_pmi_based_score(feature)
                for j,c in enumerate(self.categories):
                    matrix[i, n+(l*f_n)+j] = pmi_scores[c]
        #print(matrix)
        return csc_matrix(matrix)

############################################################
