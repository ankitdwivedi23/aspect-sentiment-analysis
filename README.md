# CS 221 Project: Aspect Based Sentiment Analysis and Summarization of Reviews

All relevant code files are in the folder named **progress**

To reproduce results, run the following command:

````
python run.py --task <task> --mode <mode> --version <version> 
--isMultiLabel <isMultiLabel> --trainFile <path to train file> 
--wordVecFile <path to word embedding weights file> --wordVecDim <dimension of word embeddings> --testFile <path to test file>

````

## Commandline Args

- --task: aspect or sentiment
- --mode: train or test
- --version: experiment version (v0, v1, v2 etc. Please see below for all versions that we tried out)
- --isMultiLabel: boolean flag to train a single multilabel classifier instead of multiple binary classifiers
- --trainFile: path to train file
- --wordVecFile: word embedding weights file
- --wordVecDim: word embedding dimension
- --testFile: path to test file

## Versions

- **v0**: Multinomial Naive Bayes using bag of words
- **v1**: Linear SVM using bag of words
- **v2**: Linear SVM using TF-IDF
- **v3**: Linear SVM using Pointwise Mutual Information (PMI)
- **v4**: Linear SVM using log count ratio (for aspect detection) or negated context TF-IDF (for sentiment analysis)
- **v5**: SimpleRNN model
- **v6**: GRU model
- **v7**: LSTM model
- **v8**: Bidirectional GRU model
- **v9**: Bidirectional LSTM model
