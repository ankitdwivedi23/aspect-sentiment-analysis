import os
import numpy as np
import pandas as pd
import math
import random
import nltk
import util
import collections
from nltk.util import ngrams
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
from scipy.sparse import hstack
#arunothia/negated-context
import nltk.sentiment.sentiment_analyzer
#=======
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# master

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

# Version-1 Feature extractor that uses scikit-learn's TfidfVectorizer
# to create ngram features with tf-idf scores as feature values and 
# custom implementation of PMI scores using scikit-learn's CountVectorizer
# To evaluate the n-gram frequencies. 
class FeatureExtractorV1(FeatureExtractor):
    def __init__(self, lexiconPath, aspect):
        self.lexiconPath = lexiconPath
        self.tfidfVectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=0.001, stop_words='english')
        self.countVectorizer = CountVectorizer(stop_words="english", analyzer='word', ngram_range=(1, 3), min_df=0.001)
        self.categories = [aspect]
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
            content = util.preprocessInput(content)
            frequencies = self.countVectorizer.fit_transform(content).sum(axis=0).tolist()[0]
            featureNames = self.countVectorizer.get_feature_names()
            d = dict(zip(featureNames, frequencies)), len(featureNames)
            #uncomment for log-count ratio
            #d = dict(zip(featureNames, frequencies)), len(content)
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
        #pmi_score = {}
        pmi_score = collections.defaultdict(float)
        frequency_context, N_context = self.getFrequencyDictionaryAndN(context)
        frequency_all, N = self.getFrequencyDictionaryAndN("all")
        for c in self.categories:
            if w in frequency_context[c] and w in frequency_all[c]:
                pmi_score[c] = math.log2( (N[c]*frequency_context[c][w])/(N_context[c]*frequency_all[c][w]) )
                #uncomment for log-count ratio
                #pmi_score[c] = (frequency_context[c][w] + 1)/(N_context[c] + 1)
            #else:
                #pmi_score[c] = 1/(N_context[c] + 1)
        return pmi_score
    
    def get_pmi_based_score(self, w):
        pmi_score_pos = self.pmi_score(w, "pos")
        pmi_score_neg = self.pmi_score(w, "neg")
        #score = {}
        score = collections.defaultdict(float)
        for c in self.categories:
            #score[c] = 0.0
            #if c in pmi_score_pos and c in pmi_score_neg:
            score[c] = (pmi_score_pos[c] - pmi_score_neg[c])
            #uncomment for log-count ratio
            #score[c] = math.log2(pmi_score_pos[c]/pmi_score_neg[c])
        return score

    def transform(self, X):        
        feature_names = self.countVectorizer.get_feature_names()
        tfidf = self.tfidfVectorizer.transform(X)
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
        #return csr_matrix(matrix)
        return hstack([csr_matrix(matrix), tfidf])

############################################################

# Version-2 Feature extractor that 
#    i. converts input to sequence of integers using Keras Tokenizer (with post padding)
#    ii. uses GloVe word embedding vectors to create an embedding matrix for input tokens
# glovePath: path to pre-trained GloVe weights - https://nlp.stanford.edu/projects/glove/
# embeddingDim: dimension of GloVe weights
class FeatureExtractorV2(FeatureExtractor):
    def __init__(self, glovePath, embeddingDim):
        self.glovePath = glovePath
        self.embeddingDim = embeddingDim
    
    def fit(self, X):
        def getEmbeddings():
            embeddings_index = {}
            with open(self.glovePath, encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
            return embeddings_index
        
        def getPaddingLength():
            word_lengths = [len(x) for x in X]
            paddingLength = math.ceil(np.percentile(word_lengths, 90))
            print("Padding Length: {}".format(paddingLength))
            return paddingLength
        
        def fitTokenizer():
            self.tokenizer_obj = Tokenizer()
            self.tokenizer_obj.fit_on_texts(X)
        
        def getEmbeddingMatrix():
            num_words = len(self.tokenizer_obj.word_index) + 1
            embedding_matrix = np.zeros((num_words, self.embeddingDim))
            word_not_found_count = 0
            for word, i in self.tokenizer_obj.word_index.items():
                if i > num_words:
                    continue
                embedding_vector = self.embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
                else:
                    word_not_found_count+=1        
            print("Word embeddings coverage: {}".format((num_words - word_not_found_count)/num_words))
            self.vocabSize = num_words
            return embedding_matrix

        self.embeddings_index = getEmbeddings()
        self.paddingLength = getPaddingLength()
        fitTokenizer()
        self.embedding_matrix = getEmbeddingMatrix()
    
    def transform(self, X):
        # pad sequences
        X_tokens =  self.tokenizer_obj.texts_to_sequences(X)
        X_tokens_pad = pad_sequences(X_tokens, maxlen=self.paddingLength, padding='post')
        return X_tokens_pad, self.embedding_matrix, self.paddingLength, self.vocabSize

############################################################

# Version-3 Feature Extractor with negated context of the n-grams along with tf-idf

class FeatureExtractorV3(FeatureExtractor):
    def __init__(self):
        self.tfidfVectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=0.001, stop_words='english')

    def addNegContextToWords(self, X):
        f = lambda x: " ".join(nltk.sentiment.util.mark_negation(x.split()))
        m = map(f, X)
        return list(m)

    def fit(self, X):
        self.tfidfVectorizer.fit([' '.join(self.addNegContextToWords(X))])
        pass

    def transform(self, X):
        return self.tfidfVectorizer.transform(self.addNegContextToWords(X))
        
