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
    
    def preprocessInput(self, X):
        return pd.Series([util.preprocessText(review) for review in X])

    def train(self, reviewsData):
        X_train = reviewsData.reviews
        X_train = self.preprocessInput(X_train)
        y_train = reviewsData.aspects
        self.featureExtractor.fit(X_train)
        featureVector= self.featureExtractor.transform(X_train)
        #print(featureVector.shape)
        #print(y_train.shape)
        self.model.fit(featureVector, y_train)
    
    def predict(self, reviewsData):
        X = reviewsData.reviews
        X = self.preprocessInput(X)
        featureVector = self.featureExtractor.transform(X)
        return self.model.predict(featureVector)
###############################################################