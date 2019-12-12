import json
from pathlib import Path

data_folder = Path("../data/")

inputPath = data_folder / 'Restaurants_Train_All.tsv'
outputPathPositive = data_folder / 'AspectLexicons' / 'Positive'
outputPathNegative = data_folder / 'AspectLexicons' / 'Negative'

aspects = all_categories = ['ambience','misc','food','price','service']

data_positive ={}
data_negative = {}

for a in aspects:
    data_positive[a] = []
    data_negative[a] = []

with open(inputPath, encoding="utf-8") as f:
    for l in f:
        line = l.split('\t')
        aspectsPresent = line[3].split(',')
        #print(aspectsPresent)
        for i,a in enumerate(aspectsPresent):
            if a == '0': 
                data_negative[aspects[i]].append(line[1])
            else:
                data_positive[aspects[i]].append(line[1])


def writeToFile(fp, text_list): 
    with open(fp, 'w+', encoding="utf-8") as f:
        for text in text_list:
            f.write(text + "\n")

for i,a in enumerate(aspects):
    writeToFile(outputPathPositive / aspects[i], data_positive[aspects[i]])
    writeToFile(outputPathNegative / aspects[i], data_negative[aspects[i]])

