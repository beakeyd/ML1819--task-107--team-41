from __future__ import division
try:
    import json
except ImportError:
    import simplejson as json
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime as dt
from sklearn import svm
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split
import pandas, numpy, textblob, string
from functools import reduce
from sklearn.neural_network import MLPClassifier

def main():
    with open('data/twitter_gender_data.json') as data:
        data = json.load(data)


        # slice data
        created_at = [d["created_at"] for d in data]
        favourites_count = [d["favourites_count"]//1000 for d in data]
        profile_background_color = [d["profile_background_color"] for d in data]
        profile_link_color = [d["profile_link_color"] for d in data]
        profile_sidebar_fill_color = [d["profile_sidebar_fill_color"] for d in data]
        profile_text_color = [d["profile_text_color"] for d in data]
        profile_sidebar_border_color = [d["profile_sidebar_border_color"] for d in data]
        listed_count = [d["listed_count"] for d in data]
        description = [d["description"] for d in data]
        tweet = [d["tweet"] for d in data]
        name = [d["name"] for d in data]
        screen_name = [d["screen_name"] for d in data]
        gender = np.where(np.array([d["gender"] for d in data]) == 'M', 0, 1)


        # create models, plot and then get accuracy of models
        created_at_acc = created_at_model(created_at, gender)
        favourites_acc = favourites_count_model(favourites_count, gender)
        color_acc = color_model(profile_background_color, profile_sidebar_fill_color,
                                profile_text_color, profile_sidebar_border_color,
                                profile_link_color, gender)
        listed_acc = listed_count_model(listed_count, gender)   
        description_acc = description_model(description, gender)
        tweet_acc = tweet_model(tweet, gender)
        name_acc = name_model(name, gender)
        #screen_name_acc = screen_name_model(screen_name, gender)

        plotAccuracy(created_at_acc, favourites_acc,
                     color_acc, listed_acc, description_acc,
                     tweet_acc, name_acc, 'Accuracy')

def normaliseData(x):
    scale=x.max(axis=0)
    return (x/scale, scale)

def plotAccuracy(created_at_acc, favourites_acc,
                 color_acc, listed_acc, description_acc,
                 tweet_acc, name_acc, graph_name):
    
    y = (created_at_acc, favourites_acc,
         color_acc, listed_acc, description_acc,
         tweet_acc, name_acc)

    X_axis = ['created_at', 'favourites',
              'color', 'listed', 'description',
              'tweet', 'name']

    y_pos = np.arange(len(X_axis))

    plt.bar(y_pos, y, align='center', alpha=0.5)
    plt.xticks(y_pos, X_axis)
    plt.ylabel('Accuracy')
    plt.title('Accuracy of features')
    graph = 'plots/' + graph_name + '.png'



def plotFeatureData(X, actY, predY, graph_name):
    fig, ax = plt.subplots(figsize=(12,2))
    ax.scatter(X, actY, label='Data', marker='+')
    ax.scatter(X, predY, label='Prediction', marker='x')

    ax.set_xlabel('Test Feature')
    ax.set_ylabel('Gender')
    ax.set_title('Feature Plot')
    graph = 'plots/' + graph_name + '.png'    
    fig.savefig(graph)

def isPredictionCorrect(y, pred):
  if y == pred:
    return 1
  else:
    return 0

def getAccuracy(actY, predY):
    acc = reduce(lambda m, n: m+n, list(map(isPredictionCorrect, actY, predY)))

    return (acc / len(actY))

''' 
    Models that ARE NOT doing text classification
'''

def created_at_model(created_at, y):
    # create Model
    (X, Xscale) = normaliseData(np.array(created_at).reshape(-1,1))

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1)

    clf = svm.SVC(kernel='linear', C = 1.0)
    clf.fit(X, y)

    # make predicitions
    predY = clf.predict(Xtest.reshape(-1, 1))
    #plot data, get and return accuracy of model
    plotFeatureData(Xtest, ytest, predY, 'Created_At')
    return getAccuracy(ytest, predY)

def color_model(profile_background_color, profile_sidebar_fill_color,
                profile_text_color, profile_sidebar_border_color,
                profile_link_color, y):
                
    (X1, _) = normaliseData(np.array([int(x, 16) for x in profile_background_color]).reshape(-1,1))
    (X2, _) = normaliseData( np.array([int(x, 16) for x in profile_sidebar_fill_color]).reshape(-1,1))
    (X3, _) = normaliseData(np.array([int(x, 16) for x in profile_text_color]).reshape(-1,1))
    (X4, _) = normaliseData(np.array([int(x, 16) for x in profile_sidebar_border_color]).reshape(-1,1))
    (X5, _)= normaliseData(np.array([int(x, 16) for x in profile_link_color]).reshape(-1,1))
    X=np.column_stack((X1, X2, X3, X4, X5))
    # create Model
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.99)

    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
    #                    hidden_layer_sizes=(5, 2), random_state=1)

    clf = svm.SVC(kernel='poly', C = 1.0, degree=3)
    clf.fit(Xtrain, ytrain)

    clf.fit(Xtrain, ytrain)

    # make predicitions
    predY = clf.predict(Xtest)
    #plot data, get and return accuracy of model

    plotFeatureData(Xtest, ytest, predY, 'Color')
    return getAccuracy(ytest, predY)

def favourites_count_model(favourites_count, y):
    # create Model
    (X, _) = normaliseData(np.array(favourites_count).reshape(-1,1))

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1)

    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
    #                    hidden_layer_sizes=(5, 2), random_state=1)
    #clf.fit(Xtrain, ytrain)

    clf = svm.SVC(kernel='poly', C = 1.0, degree=3)
    clf.fit(Xtrain, ytrain)

    # make predicitions
    predY = clf.predict(Xtest.reshape(-1,1))
    #plot data, get and return accuracy of model
    plotFeatureData(Xtest, ytest, predY, 'Favourites_Count')
    return getAccuracy(ytest, predY)

def listed_count_model(listed_count, y):
    # create Model
    (X, _) = normaliseData(np.array(listed_count).reshape(-1,1))

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1)

    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
    #                    hidden_layer_sizes=(5, 2), random_state=1)
    #clf.fit(Xtrain, ytrain)

    clf = svm.SVC(kernel='poly', C = 1.0, degree=3)
    clf.fit(Xtrain, ytrain)

    # make predicitions
    predY = clf.predict(Xtest.reshape(-1,1))
    #plot data, get and return accuracy of model
    plotFeatureData(Xtest, ytest, predY, 'Listed_Count')
    return getAccuracy(ytest, predY)

''' 
    Models that ARE doing text classification
'''

def description_model(description, gender):
    Xtest, ytest, predY = textClassification(description, gender)
    print('Description Model Metrics: ')
    print(classification_report(ytest, predY))
    return getAccuracy(ytest, predY)

def tweet_model(tweet, gender):
    Xtest, ytest, predY = textClassification(tweet, gender)
    print('Tweet Model Metrics: ')
    print(classification_report(ytest, predY))
    return getAccuracy(ytest, predY)
    
def name_model(name, gender):
    Xtest, ytest, predY = textClassification(name, gender)
    print('Name Model Metrics: ')
    print(classification_report(ytest, predY))
    return getAccuracy(ytest, predY)

def textClassification(X, y):
    # create a dataframe using texts and lables
    trainDF = pandas.DataFrame()
    trainDF['text'] = X
    trainDF['label'] = y

    # split the dataset into training and validation datasets 
    Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(trainDF['text'], trainDF['label'], test_size=0.1)

    vectorizer = CountVectorizer(stop_words='english', max_df=0.2)
    Xtrain = vectorizer.fit_transform(Xtrain)
    Xtest = vectorizer.transform(Xtest)

    model = svm.SVC(C=1.0,kernel='linear')
    model.fit(Xtrain, ytrain)

    predY = model.predict(Xtest)

    return Xtest, ytest, predY

#def combinedFeatures():


if __name__ == '__main__':
    main()
