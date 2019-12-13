from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
import nltk.sentiment.sentiment_analyzer
import util
import features
import model
import run

#import nltk
#nltk.download('punkt')

#from nltk.tokenize import word_tokenize
#from nltk.util import ngrams

##########################################################
#def get_ngrams(text, n ):
#    n_grams = ngrams(word_tokenize(text), n)
#    return [ ' '.join(grams) for grams in n_grams]

#def get_ngrams_from_review(x, k):
#    words = []
#    for i in range(1,k+1):
#        words += get_ngrams(x, i)
#    return words

def processLabels(Y):
        y_split = pd.Series([y.split(',') for y in Y])
        y_split = y_split.values.tolist()
        y_split_int = []
        for y in y_split:
            y_split_int.append([int(label) for label in y])
        return np.array(y_split_int)

###########################################################


data_folder = Path("data")
filePath = data_folder / "Restaurants_Train_All.tsv"
pmi_file_path = data_folder / "Lexicons"
aspect_pmi_file_path = data_folder / "AspectLexicons"

def readData(trainFile):
    if trainFile is not None:
        trainData = pd.read_csv(trainFile, header=None, delimiter='\t', encoding='utf-8', keep_default_na=False)
    return (trainData.iloc[:,1], processLabels(trainData.iloc[:,3]), processLabels(trainData.iloc[:,5]))


# Helper Function for Error Anaylysis: Print Feature scores for a given Review

def printFeatureScore(test, train, corpus, featureNames, weights):
    test = util.preprocessInput(test).tolist()
    #print(test)
    for i,r in enumerate(corpus):
        if r in test:
            print("*****************Review**************** " + r)
            for j,w in enumerate(featureNames):
                #print(train[i][j])
                if train[i][j] != 0:
                    print(w + " - " + str(train[i][j]))
                    print('Weight - Target - 0 - ' + str(weights[0][j]))
                    print('Weight - Target - 1 - ' + str(weights[1][j]))
                    print('Weight - Target - 2 - ' + str(weights[2][j]))
    return


#corpus = [
#    'This is good food',
#    'food is not great',
#    'bad food',
#    'Delicious food',
#    ]

#print(corpus)
#negating_lambda = lambda x: " ".join(nltk.sentiment.util.mark_negation(x.split()))
#print(list(map(negating_lambda, corpus)))

#corpus = reviews.tolist()
#print(corpus)

#X.fit(corpus)
#train = X.transform(corpus)

#X = features.FeatureExtractorV1(pmi_file_path, "food")
#X = features.FeatureExtractorV4(aspect_pmi_file_path, "food")


def tfidf_ngrams_compare():
    X = features.FeatureExtractorV0()

    (reviews, aspects, sentiments) = readData(filePath)
    data = run.ReviewsData(reviews, aspects, sentiments)
    sentimentMod = model.LinearClassifier(X, (4, "service"), "sentiment")
    corpus = sentimentMod.train(data).tolist()
    weights = sentimentMod.model.coef_

    
    featureNames = sentimentMod.featureExtractor.tfidfVectorizer.get_feature_names()
    
    # Reviews where tf-idf with 1-3 ngrams work but with only 1-gram don't work.
    # The first two are positive and the last is negative

    test = ["We have never had any problems with charging the meal or the tip, and the food was delivered quickly, but we live only a few minutes walk from them.", \
        "I go out to eat and like my courses, servers are patient and never rush courses or force another drink.", "I asked for a menu and the same waitress looked at my like I was insane."]
    printFeatureScore(test, sentimentMod.featureVectorCache["service_train"].toarray(), corpus, featureNames, weights)
    return
    
def tfidf_pmi_compare():
    X = features.FeatureExtractorV0()
    #X = features.FeatureExtractorV1(pmi_file_path, "food")

    (reviews, aspects, sentiments) = readData(filePath)
    data = run.ReviewsData(reviews, aspects, sentiments)
    sentimentMod = model.LinearClassifier(X, (2, "food"), "sentiment")
    corpus = sentimentMod.train(data).tolist()
    weights = sentimentMod.model.coef_

    featureNames = sentimentMod.featureExtractor.tfidfVectorizer.get_feature_names()
    
    # Reviews where tf-idf works but PMI does not. The first two are negative sentiments and the last two are positive

    test = ["The sauce tasted more like Chinese fast food than decent Korean.", \
        "The miso soup lacked flavor and the fish was unfortunately not as well prepared as in the past.", "My quesadilla tasted like it had been made by a three-year old with no sense of proportion or flavor.", \
        "Largest and freshest pieces of sushi, and delicious!"]
    printFeatureScore(test, sentimentMod.featureVectorCache["food_train"].toarray(), corpus, featureNames, weights)
    return

tfidf_pmi_compare()
