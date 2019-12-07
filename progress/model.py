import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
import util
import features
import keras
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D,CuDNNLSTM
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import Input, Dense, Dropout, Conv1D, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, LSTM,Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import CuDNNLSTM, CuDNNGRU

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
    
    ###############################################################

class BidirectionalGRUModel(Model):
    def __init__(self, featureExtractor, aspect, task, embeddingDim, numClasses):
        self.task = task
        self.aspect = aspect
        self.featureExtractor = featureExtractor
        self.embeddingDim = embeddingDim
        self.numClasses = numClasses

    def train(self, reviewsData):
        X_train = reviewsData.reviews
                    
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
        
        def encodeLabels():
            # encode class values as integers
            encoder = LabelEncoder()
            encoder.fit(y_train)
            encoded_y = encoder.transform(y_train)
            #convert integers to one hot encoded vectors
            y_onehot = np_utils.to_categorical(encoded_y)
            return y_onehot
        
        self.featureExtractor.fit(X_train)
        X_tokens_pad, embeddingMatrix, paddingLength, vocabSize = self.featureExtractor.transform(X_train)
        
        # Model Architecture
        inp = Input(shape=(paddingLength,))
        if type(embeddingMatrix) != type(None):
            x = Embedding(vocabSize, self.embeddingDim , weights=[embeddingMatrix],trainable=False)(inp)
        else:
            x = Embedding(vocabSize,self.embeddingDim)(inp)
        
        x = Dropout(rate=0.1)(x)
        x,x_h,x_c = Bidirectional(GRU(128, return_sequences=True,return_state=True))(x)
        max_pool = GlobalMaxPool1D()(x)
        avg_pool = GlobalAveragePooling1D()(x)
        add = concatenate([max_pool,avg_pool,x_h])
        
        x = Dense(units=512, activation="relu")(add)
        x = Dropout(rate=0.25)(x)
        x = Dense(units=self.numClasses, activation="sigmoid")(x)
        self.model = keras.models.Model(inputs=inp, outputs=x)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        y_onehot = encodeLabels()
        self.model.fit(X_tokens_pad, y_onehot, batch_size=64, epochs=25, verbose=2)
    
    def predict(self, reviewsData, mode, aspectPredictions = None):
        X = reviewsData.reviews
        X_tokens_pad, _, _, _ = self.featureExtractor.transform(X)
        y_predict = self.model.predict(X_tokens_pad)
        y_predict_label = pd.Series(y_predict.argmax(axis=1))

        if self.task == "sentiment" and mode == "test":
            all_data = pd.concat([X, pd.Series(aspectPredictions), y_predict_label], axis=1)
            all_data.columns = ["ReviewText", "AspectPred", "SentimentPred"]
            all_data.loc[all_data['AspectPred'] == 0, 'SentimentPred'] = 3
            predictions_fixed = all_data["SentimentPred"]
            return predictions_fixed            
        return y_predict_label
        

        


