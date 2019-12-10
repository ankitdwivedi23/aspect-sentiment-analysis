import os
import sys
import numpy as np
import pandas as pd
import model
import eval
import features
import pickle
import json
import collections
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from pathlib import Path

all_aspects = ['ambience','misc','food','price','service']
all_sentiments = ['negative', 'positive', 'neutral', 'na']

############################################################
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

############################################################

class Options:
    def __init__(self, task, mode, version, isMultiLabel, modelPath, outputPath, trainFile, wordVecFile, wordVecDim, testFile):
        self.task = task
        self.mode = mode
        self.version = version
        self.isMultiLabel = isMultiLabel
        self.modelPath = modelPath
        self.outputPath = outputPath
        self.trainFile = trainFile
        self.wordVecFile = wordVecFile
        self.wordVecDim = wordVecDim
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
        self.options = options
        self.trainMetricsFile = "metrics_train.json"
        self.testMetricsFile = "metrics_test.json"

    def processLabels(self, Y):
        y_split = pd.Series([y.split(',') for y in Y])
        y_split = y_split.values.tolist()
        y_split_int = []
        for y in y_split:
            y_split_int.append([int(label) for label in y])
        return np.array(y_split_int)

    def readData(self):
        print('Reading data...')
        if self.options.trainFile is not None:
            trainData = pd.read_csv(self.options.trainFile, header=None, delimiter='\t', encoding='utf-8', keep_default_na=False)
            
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

        if self.options.testFile is not None:
            testData = pd.read_csv(self.options.testFile, header=None, delimiter='\t', encoding='utf-8', keep_default_na=False)
            
            # Test Data
            X_test = testData.iloc[:,1]
            y_aspects_test = self.processLabels(testData.iloc[:,3])
            y_sentiments_test = self.processLabels(testData.iloc[:,5])

        else:
            X_train, X_test, y_aspects_train, y_aspects_test = train_test_split(X, y_aspects, test_size=0.1, random_state=23)
            _, _, y_sentiments_train, y_sentiments_test = train_test_split(X, y_sentiments, test_size=0.1, random_state=23)
        
        if self.options.mode == "train":
            self.reviewsTrain = ReviewsData(X_train, y_aspects_train, y_sentiments_train)
        self.reviewsTest = ReviewsData(X_test, y_aspects_test, y_sentiments_test)

    def loadAspectModels(self):
        if self.options.isMultiLabel:
            print('Loading aspect model...')
            self.aspectModel = joblib.load(self.options.modelPath + "/model.aspect")
        else:
            print('Loading aspect models...')
            self.aspectModels = dict()
            for (i,aspect) in enumerate(all_aspects):
                self.aspectModels[(i,aspect)] = joblib.load(self.options.modelPath + "/" + aspect + ".aspect")
    
    def loadSentimentModels(self):
        print('Loading sentiment models...')
        self.sentimentModels = dict()
        for (i,aspect) in enumerate(all_aspects):
            self.sentimentModels[(i,aspect)] = joblib.load(self.options.modelPath + "/" + aspect + ".sentiment")

    def saveAspectModels(self):
        if self.options.isMultiLabel:
            print('Saving aspect model...')
            joblib.dump(self.aspectModel, self.options.modelPath + "/model.aspect")
        else:
            print('Saving aspect models...')
            for (i,aspect) in self.aspectModels:
                joblib.dump(self.aspectModels[(i,aspect)], self.options.modelPath + "/" + aspect + ".aspect")
    
    def saveSentimentModels(self):
        print('Saving sentiment models...')
        for (i,aspect) in self.sentimentModels:
            joblib.dump(self.sentimentModels[(i,aspect)], self.options.modelPath + "/" + aspect + ".sentiment")

    def trainAspectModels(self):
        if self.options.isMultiLabel:
            aspectModelFeatureExtractor = features.FeatureExtractorV0()        
            aspectModel = None        
            print('Training aspect model...')
            aspectModel = model.OneVsRestLinearClassifier(aspectModelFeatureExtractor)
            aspectModel.train(self.reviewsTrain)
            self.aspectModel = aspectModel
        else:
            aspectModels = dict()
            lexiconsPath = Path(self.options.trainFile).parent / "AspectLexicons"
            for (i,aspect) in enumerate(all_aspects):
                print('Training aspect model for {0}...'.format(aspect))
                #aspectModelFeatureExtractor = features.FeatureExtractorV0()
                #aspectModelFeatureExtractor = features.FeatureExtractorV1(lexiconsPath, aspect)
                aspectModelFeatureExtractor = features.FeatureExtractorV2(self.options.wordVecFile, embeddingDim=self.options.wordVecDim)
                #aspectModels[(i,aspect)] = model.LinearClassifier(aspectModelFeatureExtractor, (i,aspect), self.options.task)
                aspectModels[(i,aspect)] = model.BidirectionalGRUModel(aspectModelFeatureExtractor, (i,aspect), self.options.task, embeddingDim=self.options.wordVecDim, numClasses=2)
                aspectModels[(i,aspect)].train(self.reviewsTrain)
            self.aspectModels = aspectModels
    
    def trainSentimentModels(self):
        lexiconsPath = Path(self.options.trainFile).parent / "Lexicons"
        sentimentModels = dict()
        for (i,aspect) in enumerate(all_aspects):
            print('Training sentiment model for {0}...'.format(aspect))
            #sentimentModelFeatureExtractor = features.FeatureExtractorV0()
            #sentimentModelFeatureExtractor = features.FeatureExtractorV1(lexiconsPath, aspect)
            sentimentModelFeatureExtractor = features.FeatureExtractorV2(self.options.wordVecFile, embeddingDim=self.options.wordVecDim)
            #sentimentModelFeatureExtractor = features.FeatureExtractorV3()
            #sentimentModels[(i,aspect)] = model.LinearClassifier(sentimentModelFeatureExtractor, (i,aspect), self.options.task)
            sentimentModels[(i,aspect)] = model.BidirectionalGRUModel(sentimentModelFeatureExtractor, (i,aspect), self.options.task, embeddingDim=self.options.wordVecDim, numClasses=3)
            sentimentModels[(i,aspect)].train(self.reviewsTrain)
        self.sentimentModels = sentimentModels

    def runAspectModels(self):
        if self.options.isMultiLabel:
            print('Running aspect detection model...') 
            if self.options.mode == "train":
                self.trainAspectPredictions = self.aspectModel.predict(self.reviewsTrain)
            self.testAspectPredictions = self.aspectModel.predict(self.reviewsTest)
        else:
            self.trainAspectPredictions = dict()
            self.testAspectPredictions = dict()
            for (i,aspect) in self.aspectModels:
                print('Running aspect detection model for {0}...'.format(aspect)) 
                if self.options.mode == "train":
                    self.trainAspectPredictions[(i,aspect)] = self.aspectModels[(i,aspect)].predict(self.reviewsTrain, "train")
                self.testAspectPredictions[(i,aspect)] = self.aspectModels[(i,aspect)].predict(self.reviewsTest, "test")
            
    def runSentimentModels(self):
        self.trainSentimentPredictions = dict()
        self.testSentimentPredictions = dict()
        for (i,aspect) in self.sentimentModels:
            print('Running sentiment detection model for {0}...'.format(aspect)) 
            if self.options.mode == "train":
                self.trainSentimentPredictions[(i,aspect)] = self.sentimentModels[(i,aspect)].predict(self.reviewsTrain, "train")
            
            #load aspect predictions
            filename = Path(self.options.trainFile).name + "." + aspect + ".aspect.test.output"
            #aspectPredFile = self.options.outputPath + "/" + filename
            aspectPredFile = Path(self.options.trainFile).parent / filename
            data = pd.read_csv(aspectPredFile, delimiter='\t', encoding='utf-8', keep_default_na=False)
            aspectPreds = data["AspectPred"].values
            self.testSentimentPredictions[(i,aspect)] = self.sentimentModels[(i,aspect)].predict(self.reviewsTest, "test", aspectPreds)
    
    def evalAspectModels(self):
        if self.options.isMultiLabel:
            print('Evaluating aspect model...')
            evaluator = eval.Evaluator()
            if self.options.mode == "train":
                trainMetrics = evaluator.evalPrecisionRecall(self.reviewsTrain.aspects, self.trainAspectPredictions, isMultiLabel=True)
                print('=============Aspect Detection Training Metrics==========')
                self.printEvalMetrics(trainMetrics, self.trainMetricsFile)
            testMetrics = evaluator.evalPrecisionRecall(self.reviewsTest.aspects, self.testAspectPredictions, isMultiLabel=True)
            print('=============Aspect Detection Test Metrics==============')
            self.printEvalMetrics(testMetrics, self.testMetricsFile)
        else:
            print('Evaluating aspect models...')
            evaluator = eval.Evaluator()
            if self.options.mode == "train":
                trainMetrics = evaluator.evalPrecisionRecall(self.reviewsTrain.aspects, self.trainAspectPredictions)
                print('=============Aspect Detection Training Metrics==========')
                self.printEvalMetrics(trainMetrics, self.trainMetricsFile)
            testMetrics = evaluator.evalPrecisionRecall(self.reviewsTest.aspects, self.testAspectPredictions)
            print('=============Aspect Detection Test Metrics==============')
            self.printEvalMetrics(testMetrics, self.testMetricsFile)
    
    def evalSentimentModels(self):
        print('Evaluating sentiment models...')
        evaluator = eval.Evaluator()
        if self.options.mode == "train":
            trainMetrics = evaluator.evalAccuracy(self.reviewsTrain.sentiments, self.trainSentimentPredictions, self.reviewsTrain.aspects)
            print('=============Sentiment Detection Training Metrics==========')
            self.printEvalMetrics(trainMetrics, self.trainMetricsFile)
        testMetrics = evaluator.evalAccuracy(self.reviewsTest.sentiments, self.testSentimentPredictions)
        print('=============Sentiment Detection Test Metrics==============')
        self.printEvalMetrics(testMetrics, self.testMetricsFile)

    def writeOutput(self, predictions, outputFile, reviews, isMultiLabel=False):
        aspects = [",".join([str(x) for x in labels.tolist()]) for labels in reviews.aspects]
        sentiments = [",".join([str(x) for x in labels.tolist()]) for labels in reviews.sentiments]        
        data = pd.concat([reviews.reviews, pd.Series(aspects), pd.Series(sentiments)], axis=1, ignore_index=True, sort =False)
        predictions = pd.DataFrame(predictions)

        if isMultiLabel:
            labelCount = predictions.shape[1]
            predictions[labelCount] = predictions[0].map(str)            
            for i in range(1,labelCount):
                predictions[labelCount] = predictions[labelCount] + "," + predictions[i].map(str)            
            modelOutput = pd.concat([data, predictions[labelCount]], axis=1)
        else:
            modelOutput = pd.concat([data, predictions], axis=1)
            predictedLabelColumn = "AspectPred" if self.options.task == "aspect" else "SentimentPred"
            modelOutput.columns = ["ReviewText", "Aspects", "Sentiments", predictedLabelColumn]
        modelOutput.to_csv(outputFile, sep="\t", header=True, index=False)

    def writeAspectModelOutput(self):
        if self.options.isMultiLabel:
            print('Writing aspect model output...')        
            if self.options.mode == "train":
                outputPath = self.options.outputPath + "/" + Path(self.options.trainFile).name
                self.writeOutput(self.trainAspectPredictions, outputPath + ".aspect.output", self.reviewsTrain, isMultiLabel=True)
            outputPath = self.options.outputPath + "/" + (Path(self.options.trainFile).name if self.options.testFile is None else Path(self.options.testFile).name)        
            self.writeOutput(self.testAspectPredictions, outputPath + ".aspect.test.output", self.reviewsTest, isMultiLabel=True)
        else:
            for (i,aspect) in self.aspectModels:
                if self.options.mode == "train":
                    outputPath = self.options.outputPath + "/" + Path(self.options.trainFile).name
                    self.writeOutput(self.trainAspectPredictions[(i,aspect)], outputPath + "." + aspect + ".aspect.output", self.reviewsTrain)
                outputPath = self.options.outputPath + "/" + (Path(self.options.trainFile).name if self.options.testFile is None else Path(self.options.testFile).name)
                self.writeOutput(self.testAspectPredictions[(i,aspect)], outputPath + "." + aspect + ".aspect.test.output", self.reviewsTest)

    def writeSentimentModelOutput(self):
        for (i,aspect) in self.sentimentModels:
            if self.options.mode == "train":
                outputPath = self.options.outputPath + "/" + Path(self.options.trainFile).name
                self.writeOutput(self.trainSentimentPredictions[(i,aspect)], outputPath + "." + aspect + ".sentiment.output", self.reviewsTrain)
            outputPath = self.options.outputPath + "/" + (Path(self.options.trainFile).name if self.options.testFile is None else Path(self.options.testFile).name)       
            self.writeOutput(self.testSentimentPredictions[(i,aspect)], outputPath + "." + aspect + ".sentiment.test.output", self.reviewsTest)    
    
    def printEvalMetrics(self, metrics, outputFile):
        for m in metrics:
            print("{0} = {1}".format(m, metrics[m]))
        with open(outputFile, 'w', encoding='utf-8') as jsonfile:
            json.dump(metrics, jsonfile, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    def run(self):
        self.readData()
        if self.options.task == "aspect":
            if self.options.mode == "train":
                self.trainAspectModels()
                self.saveAspectModels()
            elif self.options.mode == "test":
                self.loadAspectModels()
            else:
                print("Unknown mode")
                return
            self.runAspectModels()               
            self.evalAspectModels()
            self.writeAspectModelOutput()            
        elif self.options.task == "sentiment":
            if self.options.mode == "train":
                self.trainSentimentModels()
                self.saveSentimentModels()
            elif self.options.mode == "test":
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

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':
    task = sys.argv[1]
    mode = sys.argv[2]
    version = sys.argv[3]
    isMultiLabel = str2bool(sys.argv[4])
    modelPath = "Models"
    outputPath = "Output"
    trainFile = sys.argv[5]
    wordVecFile = None
    if len(sys.argv) > 6:
        wordVecFile = sys.argv[6]
    wordVecDim = None
    if len(sys.argv) > 7:
        wordVecDim = int(sys.argv[7])
    testFile = None
    if len(sys.argv) > 8:
        testFile = sys.argv[8]
    
    outputPath = outputPath + "/" + version
    modelPath = modelPath + "/" + version
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)

    options = Options(task, mode, version, isMultiLabel, modelPath, outputPath, trainFile, wordVecFile, wordVecDim, testFile)
    main(options)