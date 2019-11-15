import sys
import numpy as np
import pandas as pd
import model
import eval
import features
import pickle
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
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
        self.reviews = reviews.reset_index(drop=True)
        self.aspects = aspects
        self.sentiments = sentiments

############################################################

# Wrapper class to train and evaluate a model
class Runner:    
    def __init__(self, mode, modelPath, trainFile, testFile = None):
        self.mode = mode
        self.modelPath = modelPath
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
        if self.trainFile is not None:
            trainData = pd.read_csv(self.trainFile, header=None, delimiter='\t', encoding='utf-8', keep_default_na=False)
            
            # ReviewText
            X = trainData.iloc[:,1]
            # Aspect Labels
            y_aspects = self.processLabels(trainData.iloc[:,3])
            # Sentiment Labels
            y_sentiments = self.processLabels(trainData.iloc[:,5])

            # Training Data
            X_train = X
            y_aspects_train = y_aspects
            y_sentiments_train = y_sentiments

        if self.testFile is not None:
            testData = pd.read_csv(self.testFile, header=None, delimiter='\t', encoding='utf-8', keep_default_na=False)
            
            # Test Data
            X_test = testData.iloc[:,1]
            y_aspects_test = self.processLabels(testData.iloc[:,3])
            y_sentiments_test = self.processLabels(testData.iloc[:,5])

        else:
            X_train, X_test, y_aspects_train, y_aspects_test = train_test_split(X, y_aspects, test_size=0.1, random_state=23)
            _, _, y_sentiments_train, y_sentiments_test = train_test_split(X, y_sentiments, test_size=0.1, random_state=23)
        
        if self.mode == "train":
            self.reviewsTrain = ReviewsData(X_train, y_aspects_train, y_sentiments_train)
        self.reviewsTest = ReviewsData(X_test, y_aspects_test, y_sentiments_test)

    def loadModel(self):
        print('Loading model...')
        self.aspectDetectionModel = joblib.load(modelPath)
    
    def saveModel(self):
        print('Saving model...')
        joblib.dump(self.aspectDetectionModel, self.modelPath)

    def trainModel(self):
        print('Training model...')
        featureExtractor = features.FeatureExtractorV0()
        aspectDetectionModel = model.OneVsRestLinearClassifier(featureExtractor)
        aspectDetectionModel.train(self.reviewsTrain)
        self.aspectDetectionModel = aspectDetectionModel  
    
    def runModel(self):
        print('Running model...')
        aspectDetectionModel = self.aspectDetectionModel
        if self.mode == "train":
            self.trainAspectPredictions = aspectDetectionModel.predict(self.reviewsTrain)
        self.testAspectPredictions = aspectDetectionModel.predict(self.reviewsTest)        
    
    def evalModel(self):
        print('Evaluating model...')
        evaluator = eval.Evaluator()
        if self.mode == "train":
            trainMetrics = evaluator.evalAspectDetection(self.reviewsTrain.aspects, self.trainAspectPredictions)
            print('=============Aspect Detection Training Metrics==========')
            self.printEvalMetrics(trainMetrics)
        testMetrics = evaluator.evalAspectDetection(self.reviewsTest.aspects, self.testAspectPredictions)
        print('=============Aspect Detection Test Metrics==============')
        self.printEvalMetrics(testMetrics)
    
    def saveModelOutput(self):
        print('Saving model output...')
        if self.mode == "train":
            aspectPredictions = self.trainAspectPredictions
            outputFile = self.trainFile
            reviews = self.reviewsTrain
        else:
            aspectPredictions = self.testAspectPredictions
            outputFile = self.testFile
            reviews = self.reviewsTest
        
        aspects = [",".join([str(x) for x in labels.tolist()]) for labels in reviews.aspects]
        sentiments = [",".join([str(x) for x in labels.tolist()]) for labels in reviews.sentiments]        
        data = pd.concat([reviews.reviews, pd.Series(aspects), pd.Series(sentiments)], axis=1, ignore_index=True, sort =False)

        aspectPredictions = pd.DataFrame(aspectPredictions)
        
        labelCount = aspectPredictions.shape[1]
        aspectPredictions[labelCount] = aspectPredictions[0].map(str)
        for i in range(1,labelCount):
            aspectPredictions[labelCount] = aspectPredictions[labelCount] + "," + aspectPredictions[i].map(str)
        modelOutput = pd.concat([data, aspectPredictions[labelCount]], axis=1, ignore_index=True, sort =False)
        modelOutput.to_csv(outputFile + ".output", sep="\t", header=False, index=False)        
    
    def printEvalMetrics(self, metrics):
        for m in metrics:
            print("{0} = {1}".format(m, metrics[m]))

    def run(self):
        self.readData()
        if self.mode == "train":
            self.trainModel()
            self.saveModel()
        elif mode == "test":
            self.loadModel()
        else:
            print("Unknown mode")
            return
        self.runModel()
        self.evalModel()
        self.saveModelOutput()
    
############################################################

def main(mode, modelPath, trainFile, testFile):
    runner = Runner(mode, modelPath, trainFile, testFile)
    runner.run()

if __name__ == '__main__':
    mode = sys.argv[1]
    modelPath = sys.argv[2]
    trainFile = None
    testFile = None

    if mode == "train":
        trainFile = sys.argv[3]
        if len(sys.argv) > 4:
            testFile = sys.argv[4]
    elif mode == "test":
        testFile = sys.argv[3]
    
    main(mode, modelPath, trainFile, testFile)

    
