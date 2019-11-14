import sys
import numpy as np
import pandas as pd
import model
import eval
import features
from sklearn.model_selection import train_test_split
############################################################

#  Class for train/test data and labels
#
#   reviews:    pandas df/series, numpy array (or equivalent) of review texts
#   aspects:    boolean numpy array of dimension (num_examples, num_aspects), where aspects[i][j] indicates if ith review has jth aspect
#   sentiments: numpy array of dimension (num_examples, num_aspects), 
#                where sentiments[i][j] gives the sentiment of ith review for jth aspect 
#                (defaults to neutral if aspects[i][j] = 0)
class ReviewsData:
    def __init__(self, reviews, aspects, sentiments):
        self.reviews = reviews
        self.aspects = aspects
        self.sentiments = sentiments

############################################################

# Wrapper class to train and evaluate a model
class Runner:    
    def __init__(self, trainFile, testFile = None):
        self.trainFile = trainFile
        self.testFile = testFile

    def processLabels(self, Y):
        y_split = pd.Series([y.split(',') for y in Y])
        y_split = y_split.values.tolist()
        y_split_int = []
        for y in y_split:
            y_split_int.append([int(label) for label in y])
        return np.array(y_split_int)

    def readData(self):
        print('Reading data...')
        trainData = pd.read_csv(self.trainFile, header=None, delimiter='\t', encoding='utf-8', keep_default_na=False)
        
        # ReviewText
        X = trainData.iloc[:,1]
        # Aspect Labels
        y_aspects = self.processLabels(trainData.iloc[:,3])
        # Sentiment Labels
        y_sentiments = self.processLabels(trainData.iloc[:,5])

        if self.testFile is not None:
            testData = pd.read_csv(self.testFile, header=None, delimiter='\t', encoding='utf-8', keep_default_na=False)
            # Training Data
            X_train = X
            y_aspects_train = y_aspects
            y_sentiments_train = y_sentiments
            
            # Test Data
            X_test = testData.iloc[:,1]
            y_aspects_test = testData.iloc[:,3]
            y_sentiments_test = testData.iloc[:5]
        else:
            X_train, X_test, y_aspects_train, y_aspects_test = train_test_split(X, y_aspects, test_size=0.1, random_state=23)
            _, _, y_sentiments_train, y_sentiments_test = train_test_split(X, y_aspects, test_size=0.1, random_state=23)
        
        self.reviewsTrain = ReviewsData(X_train, y_aspects_train, y_sentiments_train)
        self.reviewsTest = ReviewsData(X_test, y_aspects_test, y_sentiments_test)
        
    def trainModel(self):
        print('Training model...')
        featureExtractor = features.FeatureExtractorV0()
        aspectDetectionModel = model.OneVsRestLinearClassifier(featureExtractor)
        aspectDetectionModel.train(self.reviewsTrain)
        self.aspectDetectionModel = aspectDetectionModel  
    
    def runModel(self):
        print('Running model...')
        aspectDetectionModel = self.aspectDetectionModel
        self.trainAspectPredictions = aspectDetectionModel.predict(self.reviewsTrain)
        self.testAspectPredictions = aspectDetectionModel.predict(self.reviewsTest)
    
    def evalModel(self):
        print('Evaluating model...')
        evaluator = eval.Evaluator()
        trainMetrics = evaluator.evalAspectDetection(self.reviewsTrain.aspects, self.trainAspectPredictions)
        testMetrics = evaluator.evalAspectDetection(self.reviewsTest.aspects, self.testAspectPredictions)
        print('=============Aspect Detection Training Metrics==========')
        self.printEvalMetrics(trainMetrics)
        print('=============Aspect Detection Test Metrics==============')
        self.printEvalMetrics(testMetrics)
    
    def printEvalMetrics(self, metrics):
        for m in metrics:
            print("{0} = {1}".format(m, metrics[m]))

    def run(self):
        self.readData()
        self.trainModel()
        self.runModel()
        self.evalModel()
    
############################################################

def main(trainFile, testFile):
    runner = Runner(trainFile, testFile)
    runner.run()

if __name__ == '__main__':
    trainFile = sys.argv[1]
    testFile = None
    if len(sys.argv) > 2:
        testFile = sys.argv[2]
    main(trainFile, testFile)

    
