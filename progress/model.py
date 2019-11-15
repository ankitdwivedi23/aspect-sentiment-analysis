import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
import util
import features

############################################################
# Abstract base class
class Model:
    def train(self, reviewsData):
        raise NotImplementedError("Override me")
    
    def predict(self, reviewsData):
        raise NotImplementedError("Override me")
############################################################

# Wrapper class that uses scikit-learn's OneVsRestClassifier implementation
# featureExtractor: converts input data (X) to a feature vector (numpy array or sparse matrix) 
class OneVsRestLinearClassifier(Model):
    def __init__(self, featureExtractor):
        self.model = OneVsRestClassifier(SGDClassifier())
        self.featureExtractor = featureExtractor

    def train(self, reviewsData):
        X_train = reviewsData.reviews
        X_train = util.preprocessInput(X_train)
        y_train = reviewsData.aspects
        self.featureExtractor.fit(X_train)
        featureVector= self.featureExtractor.transform(X_train)
        self.model.fit(featureVector, y_train)
    
    def predict(self, reviewsData):
        X = reviewsData.reviews
        X = util.preprocessInput(X)
        featureVector = self.featureExtractor.transform(X)
        return self.model.predict(featureVector)
###############################################################

class LinearClassifier(Model):
    def __init__(self, featureExtractor, aspect):
        self.model = SGDClassifier()
        self.featureExtractor = featureExtractor
        self.aspect = aspect

    def train(self, reviewsData):
        X_train = reviewsData.reviews
        X_train = util.preprocessInput(X_train)
        y_train = reviewsData.sentiments
        y_train = y_train[:,self.aspect[0]]
        self.featureExtractor.fit(X_train)
        print('Generating feature vector...')
        featureVector= self.featureExtractor.transform(X_train)
        print('Fitting model on training data...')
        self.model.fit(featureVector, y_train)
    
    def predict(self, reviewsData):
        X = reviewsData.reviews
        X = util.preprocessInput(X)
        featureVector = self.featureExtractor.transform(X)
        return self.model.predict(featureVector)