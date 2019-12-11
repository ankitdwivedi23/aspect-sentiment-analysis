import numpy as np
import pandas as pd

class Evaluator():
    def evalPrecisionRecall(self, truth, predicted, isMultiLabel = False, beta=1):
        totalAspects = 0
        totalPredictions = 0
        totalCorrectPredictions = 0

        if not isMultiLabel:
            predicted_all = pd.concat([pd.Series(predicted[(0,"ambience")]),pd.Series(predicted[(1,"misc")]), pd.Series(predicted[(2,"food")]), pd.Series(predicted[(3,"price")]), pd.Series(predicted[(4,"service")])], axis=1)
            predicted_all = predicted_all.values
        else:
            predicted_all = predicted

        for i in range(truth.shape[0]):
            totalAspects += sum(truth[i])
            totalPredictions += sum(predicted_all[i])
            totalCorrectPredictions += sum([x[1] for x in enumerate(predicted_all[i]) if x[1] == truth[i][x[0]]])
        
        precision = totalCorrectPredictions / totalPredictions
        recall = totalCorrectPredictions / totalAspects
        f1 = (1 + (beta ** 2)) * precision * recall / ((precision * beta ** 2) + recall) if precision > 0 and recall > 0 else 0

        metrics = dict()
        metrics["total"] = totalAspects
        metrics["predicted"] = totalPredictions
        metrics["correct"] = totalCorrectPredictions
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1

        return metrics
    
    def evalAccuracy(self, truth, predicted, filter = None):
        metrics = dict()
        allAspectsTotal = 0
        allAspectsCorrect = 0
        for aspect in predicted:
            predictedLabels = predicted[aspect].to_numpy()
            trueLabels = truth[:,aspect[0]]
            if filter is not None:
                filterLabels = filter[:,aspect[0]]
                trueLabels = trueLabels[np.where(filterLabels == 1)]
                predictedLabels = predictedLabels[np.where(filterLabels == 1)]
            print(type(predictedLabels))
            print(len(predictedLabels))
            print(type(trueLabels))
            print(len(trueLabels))

            total = len(trueLabels)
            correct = sum([1 for i in range(len(trueLabels)) if trueLabels[i] == predictedLabels[i]])
            allAspectsTotal += total
            allAspectsCorrect += correct
            accuracy = correct/total
            metrics["{0}_total".format(aspect[1])] = total
            metrics["{0}_correct".format(aspect[1])] = correct
            metrics["{0}_accuracy".format(aspect[1])] = accuracy
        
        allAspectsAccuracy = allAspectsCorrect/allAspectsTotal
        metrics["all_total"] = allAspectsTotal
        metrics["all_correct"] = allAspectsCorrect
        metrics["all_accuracy"] = allAspectsAccuracy
        
        return metrics

        





