# This script will determine which is the best technique for text classification
# So we can know which features are actually the best prediction of gender.
# Base taken from https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import Math

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

def main():
    data = json.load(data)
    
    # slice data
    description = data[:, 'description']
    gender = data[:, 'gender']

    # create a dataframe using texts and lables
    trainDF = pandas.DataFrame()
    trainDF['text'] = description
    trainDF['label'] = gender

    # split the dataset into training and validation datasets 
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

    # label encode the target variable 
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)
    
    count_vectors_acc = count_vectors(train_x, valid_x, train_y, valid_y)
    word_embeddings_acc = word_embeddings(train_x, valid_x, train_y, valid_y)
    tf_idf_acc = tf_idf(train_x, valid_x, train_y, valid_y)
    topic_models_acc = topic_models(train_x, valid_x, train_y, valid_y)

    print(str(count_vectors_acc))
    print(str(word_embeddings_acc))
    print(str(tf_idf_acc))
    print(str(topic_models_acc))
    print(str(Math.max(word_embeddings_acc, tf_idf_acc, nlp_features_acc, topic_models_acc, count_vectors_acc)))

def getAccuracy(actY, predY):
    acc = reduce(lambda m, n: m+n, list(isPredictionCorrect, actY, predY))
    return (acc / len(actY))

def count_vectors(train_x, valid_x, train_y, valid_y):
    # create a count vectorizer object 
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(trainDF['text'])

    # transform the training and validation data using count vectorizer object
    xtrain_count =  count_vect.transform(train_x)
    clf = svm.SVC(kernel='linear', C = 1.0)
    clf.fit(xtrain_count, train_y)
 
    #plot data, get and return accuracy of model
   
    xvalid_count =  count_vect.transform(valid_x)

def tf_idf(train_x, valid_x, train_y, valid_y):
    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(trainDF['text'])
    xtrain_tfidf =  tfidf_vect.transform(train_x)
    xvalid_tfidf =  tfidf_vect.transform(valid_x)

    # ngram level tf-idf 
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram.fit(trainDF['text'])
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
    xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

    # characters level tf-idf
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram_chars.fit(trainDF['text'])
    xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
    xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 

def word_embeddings(train_x, valid_x, train_y, valid_y):
    # load the pre-trained word-embedding vectors 
    embeddings_index = {}
    for i, line in enumerate(open('data/wiki-news-300d-1M.vec')):
        values = line.split()
        embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

    # create a tokenizer 
    token = text.Tokenizer()
    token.fit_on_texts(trainDF['text'])
    word_index = token.word_index

    # convert text to sequence of tokens and pad them to ensure equal length vectors 
    train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
    valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

    # create token-embedding mapping
    embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

def topic_models(train_x, valid_x, train_y, valid_y):
    
if __name__ == '__main__':
    main()