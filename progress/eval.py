class Evaluator():
    def evalAspectDetection(self, truth, predicted, beta=1):
        totalAspects = 0
        totalPredictions = 0
        totalCorrectPredictions = 0

        for i in range(truth.shape[0]):
            totalAspects += sum(truth[i])
            totalPredictions += sum(predicted[i])
            totalCorrectPredictions += sum([x[1] for x in enumerate(predicted[i]) if x[1] == truth[i][x[0]]])
        
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
    
    def evalSentimentDetection(self, truth, predicted):
        metrics = dict()
        allAspectsTotal = 0
        allAspectsCorrect = 0
        for aspect in predicted:
            predictedLabels = predicted[aspect]
            trueLabels = truth[:,aspect[0]]
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

        





