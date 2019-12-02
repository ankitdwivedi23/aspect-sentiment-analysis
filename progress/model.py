import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
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
    def __init__(self, featureExtractor, aspect, task):
        self.model = SGDClassifier()
        self.featureExtractor = featureExtractor
        self.aspect = aspect
        self.task = task
        self.featureVectorCache = dict()

    def train(self, reviewsData):
        X_train = reviewsData.reviews
        X_train = util.preprocessInput(X_train)
        
        if self.task == "aspect":
            y_train = reviewsData.aspects
            y_train = y_train[:,self.aspect[0]]
        elif self.task == "sentiment":
            y_train = reviewsData.sentiments
            y_train = pd.Series(y_train[:,self.aspect[0]])
            all_data = pd.concat([X_train,y_train], axis=1)
            all_data.columns = ["ReviewText", "Sentiment"]
            all_data = all_data[all_data["Sentiment"] != 3].reset_index(drop=True)
            X_train = all_data["ReviewText"]
            y_train = all_data["Sentiment"]
        self.featureExtractor.fit(X_train)
        print('Generating feature vector...')
        cacheKey = self.aspect[1] + "_train"
        if cacheKey not in self.featureVectorCache:
            self.featureVectorCache[cacheKey] = self.featureExtractor.transform(X_train)
        featureVector= self.featureVectorCache[cacheKey]
        print('Fitting model on training data...')
        self.model.fit(featureVector, y_train)
    
    def predict(self, reviewsData, mode, aspectPredictions = None):
        X = reviewsData.reviews
        X = util.preprocessInput(X)
        cacheKey = self.aspect[1] + "_" + mode
        if cacheKey not in self.featureVectorCache:
            self.featureVectorCache[cacheKey] = self.featureExtractor.transform(X)        
        featureVector = self.featureVectorCache[cacheKey]
        predictions = self.model.predict(featureVector)
        
        if self.task == "sentiment" and mode == "test":
            all_data = pd.concat([X, pd.Series(aspectPredictions), pd.Series(predictions)], axis=1)
            all_data.columns = ["ReviewText", "AspectPred", "SentimentPred"]
            all_data.loc[all_data['AspectPred'] == 0, 'SentimentPred'] = 3
            predictions_fixed = all_data["SentimentPred"]
            return predictions_fixed
        
        return predictions

        
