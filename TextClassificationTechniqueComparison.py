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
