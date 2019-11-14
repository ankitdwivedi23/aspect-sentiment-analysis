import nltk
from nltk.stem import WordNetLemmatizer
import re

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer() 

def lemmatize(text):
    lemmedTokens = []
    for token in text.split():
        lemmedTokens.append(lemmatizer.lemmatize(token))
    return ' '.join(lemmedTokens)

def preprocessText(text):
    # remove special characters
    text = re.sub(r'\W', ' ', text)    
    # replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    text = lemmatize(text)
    return text

    