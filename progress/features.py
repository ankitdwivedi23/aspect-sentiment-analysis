import numpy as np
import pandas as pd
import math
import random
import nltk
from nltk.util import ngrams
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csc_matrix
from scipy.sparse import dok_matrix
from scipy.sparse import hstack

###########################################################
# Helper Functions

def getPositivePath(lexiconPath, p):
    return Path(lexiconPath) / "Positive" / (p)

def getNegativePath(lexiconPath, p):
        return Path(lexiconPath) / "Negative" / (p)

def getPath(lexiconPath, p, context):
    if context == "pos":
        return getPositivePath(lexiconPath, p)
    else:
        return getNegativePath(lexiconPath, p)
    
def getFileContent(fp):
    with open(fp, encoding='utf-8') as f:
        content = f.readlines()
    #return random.sample(content, int(0.001*len(content)))
    return content

def extract_ngrams(data, num):
    n_grams = ngrams(data.split(), num)
    return [ ' '.join(grams) for grams in n_grams]

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
    def __init__(self, lexiconPath, aspect):
        self.lexiconPath = lexiconPath
        #self.tfidfVectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=0.001, stop_words='english')
        self.countVectorizer = CountVectorizer(stop_words="english", analyzer='word', ngram_range=(1, 3), min_df=0.001)
        self.categories = [aspect]
        self.positive_frequencies = {}
        self.negative_frequencies = {}
        self.all_frequencies = {}
        self.N_pos = {}
        self.N_neg = {}
        self.N = {}

    def fit(self, X):
        #self.tfidfVectorizer.fit([' '.join(X)])
        
        def getFrequencyDictionaryAndN(filePath):
            content = getFileContent(filePath)
            frequencies = self.countVectorizer.fit_transform(content).sum(axis=0).tolist()[0]
            featureNames = self.countVectorizer.get_feature_names()
            d = dict(zip(featureNames, frequencies)), len(featureNames)
            return d

        for c in self.categories:
            self.positive_frequencies[c], self.N_pos[c] = getFrequencyDictionaryAndN(getPath(self.lexiconPath, c, "pos"))
            self.negative_frequencies[c], self.N_neg[c] = getFrequencyDictionaryAndN(getPath(self.lexiconPath, c, "neg"))
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
        feature_names = self.countVectorizer.get_feature_names()
        #feature_names = self.tfidfVectorizer.get_feature_names()
        #tfidf = self.tfidfVectorizer.transform(X)
        #print(feature_names)
        #m,n = tfidf.toarray().shape
        m = len(X)
        feature_num_names = {k:v for v,k in enumerate(feature_names)}
        #print("(m,n) = ", (m,n))
        l,f = len(self.categories), len(feature_names)
        print("(l,f) = ", (l,f))
        matrix = dok_matrix((m,f*l))
        print("dok_matrix: ", matrix.shape)
        for i in range(m):
            #print(X[i])
            grams = extract_ngrams(X[i], 1) + extract_ngrams(X[i], 2) + extract_ngrams(X[i], 3)
            for feature in grams:
                if feature not in feature_names:
                    continue
                f_n = feature_num_names[feature]
                pmi_scores = self.get_pmi_based_score(feature)
                for j,c in enumerate(self.categories):
                    matrix[i, (l*f_n)+j] = 0
                    if c in pmi_scores:
                        matrix[i, (l*f_n)+j] = pmi_scores[c]
        return csc_matrix(matrix)
        #return csc_matrix(hstack((matrix, tfidf)))

############################################################
