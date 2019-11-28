import sys
import numpy as np
import pandas as pd
import model
import eval
import features
import pickle
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from pathlib import Path

all_aspects = ['ambience','misc','food','price','service']
all_sentiments = ['negative', 'positive', 'neutral', 'conflict', 'na']

############################################################

class Options:
    def __init__(self, task, mode, modelPath, trainFile, testFile):
        self.task = task
        self.mode = mode
        self.modelPath = modelPath
        self.trainFile = trainFile
        self.testFile = testFile

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
    def __init__(self, options):
        self.task = options.task
        self.mode = options.mode
        self.modelPath = options.modelPath
        self.trainFile = options.trainFile
        self.testFile = options.testFile

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

    def loadAspectModels(self):
        #print('Loading aspect model...')
        #self.aspectModel = joblib.load(self.modelPath)
        print('Loading aspect models...')
        self.aspectModels = dict()
        for (i,aspect) in enumerate(all_aspects):
            self.aspectModels[(i,aspect)] = joblib.load(self.modelPath + "." + aspect)
    
    def loadSentimentModels(self):
        print('Loading sentiment models...')
        self.sentimentModels = dict()
        for (i,aspect) in enumerate(all_aspects):
            self.sentimentModels[(i,aspect)] = joblib.load(self.modelPath + "." + aspect)

    def saveAspectModels(self):
        #print('Saving aspect model...')
        #joblib.dump(self.aspectModel, self.modelPath)
        print('Saving aspect models...')
        for (i,aspect) in self.aspectModels:
            joblib.dump(self.aspectModels[(i,aspect)], self.modelPath + "." + aspect)
    
    def saveSentimentModels(self):
        print('Saving sentiment models...')
        for (i,aspect) in self.sentimentModels:
            joblib.dump(self.sentimentModels[(i,aspect)], self.modelPath + "." + aspect)

    def trainAspectModels(self):
        #aspectModelFeatureExtractor = features.FeatureExtractorV2()        
        #aspectModel = None        
        #print('Training aspect model...')
        #aspectModel = model.OneVsRestLinearClassifier(aspectModelFeatureExtractor)
        #aspectModel.train(self.reviewsTrain)
        #self.aspectModel = aspectModel

        aspectModels = dict()
        aspectModelFeatureExtractor = features.FeatureExtractorV0()
        for (i,aspect) in enumerate(all_aspects):
            print('Training aspect model for {0}...'.format(aspect))
            aspectModels[(i,aspect)] = model.LinearClassifier(aspectModelFeatureExtractor, (i,aspect), self.task)
            aspectModels[(i,aspect)].train(self.reviewsTrain)
        self.aspectModels = aspectModels
    
    def trainSentimentModels(self):
        trainPath = Path(self.trainFile).parent / "Lexicons"
        sentimentModels = dict()
        #sentimentModelFeatureExtractor = features.FeatureExtractorV0()
        for (i,aspect) in enumerate(all_aspects):
            print('Training sentiment model for {0}...'.format(aspect))
            sentimentModelFeatureExtractor = features.FeatureExtractorV1(trainPath, aspect)
            sentimentModels[(i,aspect)] = model.LinearClassifier(sentimentModelFeatureExtractor, (i,aspect), self.task)
            sentimentModels[(i,aspect)].train(self.reviewsTrain)
        self.sentimentModels = sentimentModels

    def runAspectModels(self):
        #print('Running aspect detection model...') 
        #if self.mode == "train":
        #    self.trainAspectPredictions = self.aspectModel.predict(self.reviewsTrain)
        #self.testAspectPredictions = self.aspectModel.predict(self.reviewsTest)

        self.trainAspectPredictions = dict()
        self.testAspectPredictions = dict()
        for (i,aspect) in self.aspectModels:
            print('Running aspect detection model for {0}...'.format(aspect)) 
            if self.mode == "train":
                self.trainAspectPredictions[(i,aspect)] = self.aspectModels[(i,aspect)].predict(self.reviewsTrain, "train")
            self.testAspectPredictions[(i,aspect)] = self.aspectModels[(i,aspect)].predict(self.reviewsTest, "test")
            
    def runSentimentModels(self):
        self.trainSentimentPredictions = dict()
        self.testSentimentPredictions = dict()
        for (i,aspect) in self.sentimentModels:
            print('Running sentiment detection model for {0}...'.format(aspect)) 
            if self.mode == "train":
                self.trainSentimentPredictions[(i,aspect)] = self.sentimentModels[(i,aspect)].predict(self.reviewsTrain, "train")
            self.testSentimentPredictions[(i,aspect)] = self.sentimentModels[(i,aspect)].predict(self.reviewsTest, "test")
    
    def evalSentimentModels(self):
        print('Evaluating sentiment models...')
        evaluator = eval.Evaluator()
        if self.mode == "train":
            trainMetrics = evaluator.evalSentimentDetection(self.reviewsTrain.sentiments, self.trainSentimentPredictions)
            print('=============Sentiment Detection Training Metrics==========')
            self.printEvalMetrics(trainMetrics)
        testMetrics = evaluator.evalSentimentDetection(self.reviewsTest.sentiments, self.testSentimentPredictions)
        print('=============Sentiment Detection Test Metrics==============')
        self.printEvalMetrics(testMetrics)

    def evalAspectModels(self):
        #print('Evaluating aspect model...')
        #evaluator = eval.Evaluator()
        #if self.mode == "train":
        #    trainMetrics = evaluator.evalAspectDetection(self.reviewsTrain.aspects, self.trainAspectPredictions)
        #    print('=============Aspect Detection Training Metrics==========')
        #    self.printEvalMetrics(trainMetrics)
        #testMetrics = evaluator.evalAspectDetection(self.reviewsTest.aspects, self.testAspectPredictions)
        #print('=============Aspect Detection Test Metrics==============')
        #self.printEvalMetrics(testMetrics)

        print('Evaluating sentiment models...')
        evaluator = eval.Evaluator()
        if self.mode == "train":
            trainMetrics = evaluator.evalSentimentDetection(self.reviewsTrain.aspects, self.trainAspectPredictions)
            print('=============Aspect Detection Training Metrics==========')
            self.printEvalMetrics(trainMetrics)
        testMetrics = evaluator.evalSentimentDetection(self.reviewsTest.aspects, self.testAspectPredictions)
        print('=============Aspect Detection Test Metrics==============')
        self.printEvalMetrics(testMetrics)
    
    def writeOutput(self, predictions, outputFile, reviews):
        aspects = [",".join([str(x) for x in labels.tolist()]) for labels in reviews.aspects]
        sentiments = [",".join([str(x) for x in labels.tolist()]) for labels in reviews.sentiments]        
        data = pd.concat([reviews.reviews, pd.Series(aspects), pd.Series(sentiments)], axis=1, ignore_index=True, sort =False)
        predictions = pd.DataFrame(predictions)

        labelCount = predictions.shape[1]
        predictions[labelCount] = predictions[0].map(str)
            
        for i in range(1,labelCount):
            predictions[labelCount] = predictions[labelCount] + "," + predictions[i].map(str)
            
        modelOutput = pd.concat([data, predictions[labelCount]], axis=1, ignore_index=True, sort =False)
        modelOutput.to_csv(outputFile, sep="\t", header=False, index=False)

    def writeAspectModelOutput(self):
        #print('Writing aspect model output...')        
        #if self.mode == "train":
        #    self.writeOutput(self.trainAspectPredictions, self.trainFile + ".output", self.reviewsTrain)        
        #self.writeOutput(self.testAspectPredictions, self.trainFile + ".test.output" if self.testFile is None else self.testFile + ".output", self.reviewsTest)
        for (i,aspect) in self.aspectModels:
            if self.mode == "train":
                self.writeOutput(self.trainAspectPredictions[(i,aspect)], self.trainFile + "." + aspect + ".aspect.output", self.reviewsTrain)        
            self.writeOutput(self.testAspectPredictions[(i,aspect)], self.trainFile + "." + aspect + ".aspect.test.output" if self.testFile is None else self.testFile + ".aspect.output", self.reviewsTest)

    def writeSentimentModelOutput(self):
        for (i,aspect) in self.sentimentModels:
            if self.mode == "train":
                self.writeOutput(self.trainSentimentPredictions[(i,aspect)], self.trainFile + "." + aspect + ".output", self.reviewsTrain)        
            self.writeOutput(self.testSentimentPredictions[(i,aspect)], self.trainFile + "." + aspect + ".test.output" if self.testFile is None else self.testFile + ".output", self.reviewsTest)    
    
    def printEvalMetrics(self, metrics):
        for m in metrics:
            print("{0} = {1}".format(m, metrics[m]))    

    def run(self):
        self.readData()
        if self.task == "aspect":
            if self.mode == "train":
                self.trainAspectModels()
                self.saveAspectModels()
            elif mode == "test":
                self.loadAspectModels()
            else:
                print("Unknown mode")
                return
            self.runAspectModels()               
            self.evalAspectModels()
            self.writeAspectModelOutput()            
        elif self.task == "sentiment":
            if self.mode == "train":
                self.trainSentimentModels()
                self.saveSentimentModels()
            elif mode == "test":
                self.loadSentimentModels()
            else:
                print("Unknown mode")
                return
            self.runSentimentModels()               
            self.evalSentimentModels()
            self.writeSentimentModelOutput() 
    
############################################################

def main(options):
    runner = Runner(options)
    runner.run()

if __name__ == '__main__':
    task = sys.argv[1]
    mode = sys.argv[2]
    modelPath = sys.argv[3]
    trainFile = sys.argv[4]
    testFile = None
    if len(sys.argv) > 5:
        testFile = sys.argv[5]    
    options = Options(task, mode, modelPath, trainFile, testFile)
    main(options)

    
