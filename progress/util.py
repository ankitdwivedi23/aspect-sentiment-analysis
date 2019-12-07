import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
import re
from autocorrect import Speller

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
spell = Speller(lang='en')

def lemmatize(text):
    lemmedTokens = []
    for token in text.split():
        lemmedTokens.append(lemmatizer.lemmatize(token))
    return ' '.join(lemmedTokens)

def preprocessText(text):
    # replace n't with not
    text = re.sub(r'n\'t', ' not ', text)
    # add space in front of .;
    text = re.sub(r'\.', ' . ', text)
    # remove special characters
    text = re.sub(r'[&%*#@^$!\_\-\'()?0-9\;\,]', ' ', text)    
    # replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    #text = lemmatize(text)
    return spell(text)

def preprocessInput(X):
        return pd.Series([preprocessText(x) for x in X])

    
