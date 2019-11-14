class Evaluator():
    def evalAspectDetection(self, correct, predicted, beta=1):
        totalAspects = 0
        totalPredictions = 0
        totalCorrectPredictions = 0

        for i in range(correct.shape[0]):
            totalAspects += sum(correct[i])
            totalPredictions += sum(predicted[i])
            totalCorrectPredictions += sum([x[1] for x in enumerate(predicted[i]) if x[1] == correct[i][x[0]]])
        
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

        





