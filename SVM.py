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
from sklearn.model_selection import train_test_split
from functools import reduce
from sklearn.neural_network import MLPClassifier


def main():
    with open('data/twitter_gender_data.json') as data:
        data = json.load(data)

        # slice data
        created_at = [d["created_at"] for d in data]
        favourites_count = [d["favourites_count"] for d in data]
        profile_background_color = [d["profile_background_color"] for d in data]
        listed_count = [d["listed_count"] for d in data]
        description = [d["description"] for d in data]
        tweet = [d["tweet"] for d in data]
        name = [d["name"] for d in data]
        screen_name = [d["screen_name"] for d in data]
        gender = [d["gender"] for d in data]

        # create models, plot and then get accuracy of models
        created_at_acc = created_at_model(created_at, gender)
        favourites_acc = favourites_count_model(favourites_count, gender)
        color_acc = color_model(profile_background_color, gender)
        listed_acc = listed_count_model(listed_count, gender)
        print('listed accuracy:')
        print(str(listed_acc))
        print('color accuracy:')
        print(str(color_acc))
        print('favourites accuracy:')
        print(str(favourites_acc))
        #description_acc = description(description, gender)
        #tweet_acc = tweet(tweet, gender)
        #name_acc = name(name, gender)
        #screen_name_acc = screen_name(screen_name, gender)

        #scikit_test()

        #plotAccuracy(created_at_acc, favourites_acc,
        #             color_acc, listed_acc, description_acc,
        #             tweet_acc, name_acc, screen_name_acc, 'Accuracy')

def normaliseData(x):
    scale=x.max(axis=0)
    return (x/scale, scale)

def scikit_test():
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    clf = svm.SVC(gamma='scale')
    clf.fit(X, y)

def plotAccuracy(created_at_acc, favourites_acc,
                 color_acc, listed_acc, description_acc,
                 tweet_acc, name_acc, screen_name_acc, graph_name):
    

    X = ['created_at', 'favourites',
         'color', 'listed', 'description',
         'tweet', 'name', 'screen_name']
    y = [created_at_acc, favourites_acc,
         color_acc, listed_acc, description_acc,
         tweet_acc, name_acc, screen_name_acc]

    fig, ax = plt.bar(figsize=(12,2))
    ax.bar(X, y, label='Accuracy')

    ax.set_xlabel('Feature')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy of Features chart')
    graph = 'plots/' + graph_name + '.png'   
    fig.savefig(graph)


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

def created_at_model(created_at, gender):
    # create Model
    X = np.array(created_at)
    y = np.array(gender)

    date = dt.strptime(X[0], "%a %b %d %H:%M:%S +0000 %Y")
    print(time.mktime(date.timetuple()))

    #clf = svm.SVC(kernel='linear', C = 1.0)
    #clf.fit(X, y)

    # make predicitions
    #predY = [clf.predict(x) for x in X]
    #plot data, get and return accuracy of model
    #plotFeatureData(X, y, predY, 'Created_At')
    # validate(X, y, predY)
    return 0 

def color_model(profile_background_color, gender):
                
    (X, _) = normaliseData(np.array([int(x, 16) for x in profile_background_color]).reshape(-1,1))

    # create Model
    #(X, _) = normaliseData(np.array(newList).reshape(-1,1))
    y = np.where(np.array(gender) == 'M', 0, 1)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)

    print(Xtrain)
    print(ytrain)
    
    clf.fit(Xtrain, ytrain)

    print('here!')

    # make predicitions
    predY = clf.predict(Xtest.reshape(-1, 1))
    #plot data, get and return accuracy of model
    print(predY)
    plotFeatureData(Xtest, ytest, predY, 'Color')
    return getAccuracy(ytest, predY)

def favourites_count_model(favourites_count, gender):
    # create Model
    (X, Xscale) = normaliseData(np.array(favourites_count).reshape(-1,1))
    y = np.where(np.array(gender) == 'M', 0, 1)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(Xtrain, ytrain)

    # make predicitions
    predY = clf.predict(Xtest.reshape(-1,1))
    #plot data, get and return accuracy of model
    plotFeatureData(Xtest, ytest, predY, 'Favourites_Count')
    return getAccuracy(ytest, predY)

def listed_count_model(listed_count, gender):
    # create Model
    (X, Xscale) = normaliseData(np.array(listed_count).reshape(-1,1))
    y = np.where(np.array(gender) == 'M', 0, 1)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(Xtrain, ytrain)

    #clf = svm.SVC(kernel='poly', C = 1.0, degree=10)
    #clf.fit(Xtrain, ytrain)

    # make predicitions
    predY = clf.predict(Xtest.reshape(-1,1))
    #plot data, get and return accuracy of model
    plotFeatureData(Xtest, ytest, predY, 'Listed_Count')
    return getAccuracy(ytest, predY)

''' 
    Models that ARE doing text classification
'''

#def description():
#def tweet():
#def name():
#def screen_name():


#def combinedFeatures():

if __name__ == '__main__':
    main()
