import json
from pathlib import Path

data_folder = Path("../data/")

inputPath = data_folder / 'Restaurants_Train.tsv'
outputPathPositive = data_folder / 'Lexicons' / 'Positive'
outputPathNegative = data_folder / 'Lexicons' / 'Negative'
outputPathNeutral = data_folder / 'Lexicons' / 'Neutral'

aspects = all_categories = ['ambience','misc','food','price','service']

data_positive ={}
data_negative = {}
data_neutral = {}

for a in aspects:
    data_positive[a] = []
    data_negative[a] = []
    data_neutral[a] = []

with open(inputPath, encoding="utf-8") as f:
    for l in f:
        line = l.split('\t')
        aspectsPresent = line[3].split(',')
        sentiment = line[5].split(',')
        sentiment[4] = sentiment[4].replace('\n', '')
        #print(aspectsPresent)
        #print(sentiment)
        for i,a in enumerate(aspectsPresent):
            if sentiment[i] == '0': 
                data_negative[aspects[i]].append(line[1])
            elif sentiment[i] == '1':
                data_positive[aspects[i]].append(line[1])
            else:
                data_neutral[aspects[i]].append(line[1])


def writeToFile(fp, text_list): 
    with open(fp, 'w+', encoding="utf-8") as f:
        for text in text_list:
            f.write(text + "\n")

for i,a in enumerate(aspects):
    writeToFile(outputPathPositive / aspects[i], data_positive[aspects[i]])
    writeToFile(outputPathNegative / aspects[i], data_negative[aspects[i]])
    writeToFile(outputPathNeutral / aspects[i], data_neutral[aspects[i]])

