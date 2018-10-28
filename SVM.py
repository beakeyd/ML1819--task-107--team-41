from __future__ import division
try:
    import json
except ImportError:
    import simplejson as json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from mpl_toolkits.mplot3d import Axes3D
import time
from datetime import datetime as dt
from sklearn import svm
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
import pandas, numpy, textblob, string
from functools import reduce
from sklearn.neural_network import MLPClassifier

def main():
    with open('data/twitter_gender_data.json') as data:
        data = json.load(data)

        # slice data
        created_at = [d["created_at"] for d in data]
        favourites_count = [d["favourites_count"]for d in data]
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

'''
        # create models, plot and then get accuracy of models
        created_at_acc = created_at_model(created_at, gender)
        favourites_acc = favourites_count_model(favourites_count, gender)
        color_acc = color_model(profile_background_color, profile_sidebar_fill_color,
                                profile_text_color,
                                profile_link_color, gender)
        listed_acc = listed_count_model(listed_count, gender)   
        description_acc = description_model(description, gender)
        tweet_acc = tweet_model(tweet, gender)
        name_acc = name_model(name, gender)

        plotAccuracy(created_at_acc, favourites_acc,
                     color_acc, listed_acc, description_acc,
                     tweet_acc, name_acc, 'Accuracy')
'''
        combinedFeatures(name, created_at, gender)

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

def plotSingleFeatureData(X, actY, predY, graph_name):
    fig, ax = plt.subplots(figsize=(12,2))
    ax.scatter(X, actY, label='Data', marker='+')
    ax.scatter(X, predY, label='Prediction', marker='x')

    ax.set_xlabel('Test Feature')
    ax.set_ylabel('Gender')
    ax.set_title('Feature Plot')
    graph = 'plots/' + graph_name + '.png'    
    fig.savefig(graph)

def plotMultiFeatureData(X, y, predY, scale, model, graph_name):
    fig, ax = plt.subplots(figsize=(12,8))
    # plot the data
    positive = y > 0
    negative = y < 0
    ax.scatter(X[:, 0], X[:, 1], c='b', marker='o', label='')
    ax.scatter(X[:, 0], X[:, 1], c='r', marker='x', label='')
    # calc the decision boundary
    x_min, x_max = X.min() - 0.1, X.max() + 0.1
    y_min, y_max = y.min() - 0.1, y.max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
    Z = model.predict(np.column_stack((np.ones(xx.ravel().shape),xx.ravel(), yy.ravel(), np.square(xx.ravel()))))
    Z = Z.reshape(xx.shape)
    ax.contour(xx*Xscale[1], yy*Xscale[2], Z)
    ax.legend()
    ax.set_xlabel('Color 1')
    ax.set_ylabel('Color 2')
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
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)
    clf = svm.SVC(kernel='linear', C = 1.0)
    #getCAndGamma(clf, X, y, 'created_at_model')
    clf.fit(X, y)

    # make predicitions
    predY = clf.predict(Xtest.reshape(-1, 1))
    #plot data, get and return accuracy of model
    print('created_at Model metrics: ')
    print(classification_report(ytest, predY))
    plotSingleFeatureData(Xtest, ytest, predY, 'Created_At')

    return getAccuracy(ytest, predY)

def color_model(profile_background_color, profile_sidebar_fill_color,
                profile_text_color, profile_sidebar_border_color, gender):
    (X1, scale) = normaliseData(np.array([int(x, 16) for x in profile_background_color]).reshape(-1,1))
    (X2, _) = normaliseData( np.array([int(x, 16) for x in profile_sidebar_fill_color]).reshape(-1,1))
    (X3, _) = normaliseData(np.array([int(x, 16) for x in profile_text_color]).reshape(-1,1))
    (X4, _) = normaliseData(np.array([int(x, 16) for x in profile_sidebar_border_color]).reshape(-1,1))
    X=np.column_stack((X1, X2))
    # create Model
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, gender, test_size=0.1)
    clf = svm.SVC(kernel='poly', C = 1, degree=5)
    #getCAndGamma(clf, Xtest, ytest, 'color_model')
    clf.fit(Xtrain, ytrain)

    # make predicitions
    predY = clf.predict(Xtest)
    #plot data, get and return accuracy of model
    print('color Model metrics: ')
    print(classification_report(ytest, predY))
    #plotMultiFeatureData(Xtest, ytest, predY, scale, clf, 'Color')
    return getAccuracy(ytest, predY)    

def favourites_count_model(favourites_count, y):
    # create Model
    (X, _) = normaliseData(np.array(favourites_count).reshape(-1,1))
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)
    clf = svm.SVC(kernel='poly', C=10, gamma=10, degree=3)
    #getCAndGamma(clf, X, y, 'favourites_count_model')
    clf.fit(Xtrain, ytrain)
    # make predicitions
    predY = clf.predict(Xtest.reshape(-1,1))
    #plot data, get and return accuracy of model
    print('favourites_count Model metrics: ')
    print(classification_report(ytest, predY))
    plotSingleFeatureData(Xtest, ytest, predY, 'Favourites_Count')

    return getAccuracy(ytest, predY)

def listed_count_model(listed_count, y):
    # create Model
    (X, _) = normaliseData(np.array(listed_count).reshape(-1,1))
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)
    clf = svm.SVC(kernel='poly', C=10, degree=3)
    #getCAndGamma(clf, X, y, 'listed_count_model')
    clf.fit(Xtrain, ytrain)

    # make predicitions
    predY = clf.predict(Xtest.reshape(-1,1))
    #plot data, get and return accuracy of model
    print('listed_count Model metrics: ')
    print(classification_report(ytest, predY))
    plotSingleFeatureData(Xtest, ytest, predY, 'Listed_Count')
    return getAccuracy(ytest, predY)

''' 
    Models that ARE doing text classification
'''

def description_model(description, gender):
    Xtest, ytest, predY = textClassification(description, gender,  'description_model')
    print('Description Model Metrics: ')
    print(classification_report(ytest, predY))
    return getAccuracy(ytest, predY)

def tweet_model(tweet, gender):
    Xtest, ytest, predY = textClassification(tweet, gender, 'tweet_model')
    print('Tweet Model Metrics: ')
    print(classification_report(ytest, predY))
    return getAccuracy(ytest, predY)
    
def name_model(name, gender):
    Xtest, ytest, predY = textClassification(name, gender, 'name_model')
    print('Name Model Metrics: ')
    print(classification_report(ytest, predY))
    return getAccuracy(ytest, predY)

def textClassification(X, y, name):
    # create a dataframe using texts and lables
    trainDF = pandas.DataFrame()
    trainDF['text'] = X
    trainDF['label'] = y

    # split the dataset into training and validation datasets 
    Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(trainDF['text'], trainDF['label'], test_size=0.3)
    vectorizer = CountVectorizer(stop_words='english', max_df=0.2)
    Xtrain = vectorizer.fit_transform(Xtrain)
    Xtest = vectorizer.transform(Xtest)

    model = svm.SVC(C=10, gamma=1, kernel='poly')
  #  getCAndGamma(model, X, y, name)
    model.fit(Xtrain, ytrain)
    predY = model.predict(Xtest)
    return Xtest, ytest, predY

'''
def getCAndGamma(model, X, y, name):
    C_s, gamma_s = np.meshgrid(np.logspace(-2, 1, 20), np.logspace(-2, 1, 20))
    print(name)
    scores = list()
    i=0; j=0
    z = 0
    for C, gamma in zip(C_s.ravel(),gamma_s.ravel()):
        print(str(z))
        z = z + 1
        model.C = C
        model.gamma = gamma
        this_scores = cross_val_score(model, X, y, cv=5)
        scores.append(np.mean(this_scores))
    
    scores=np.array(scores)
    scores=scores.reshape(C_s.shape)
    fig2, ax2 = plt.subplots(figsize=(20,20))
    c=ax2.contourf(C_s,gamma_s,scores)
    ax2.set_xlabel('C')
    ax2.set_ylabel('gamma')
    fig2.colorbar(c)
    graph = 'plots/' + name + '.png'
    fig2.savefig(graph)
'''

def combinedFeatures(feature1_data, feature2_data, y):
        # perform train/test split
        Xtrain1, Xtest1, ytrain1, ytest1 = model_selection.train_test_split(feature1_data, y, test_size=0.1)
        Xtrain2, Xtest2, ytrain2, ytest2 = model_selection.train_test_split(feature2_data, y, test_size=0.1)

      
        # define text pre-processing pipeline
        text_pipeline = Pipeline([
            ('vect', vectorizer=CountVectorizer(Xtrain1))
        ])
        
        # define pipeline for pre-processing of numeric features
        numeric_pipeline = Pipeline([
            ('scaler', MinMaxScaler(Xtrain2))
        ])

        # combine both steps into a single pipeline
        pipeline = Pipeline([
            ('features', FeatureUnion([
                ('text_processing', text_pipeline),
                ('num_processing', numeric_pipeline)
            ])),
            ('clf',SGDClassifier(loss='hinge', penalty='l2',
                                               alpha=1e-3, random_state=42,
                                               max_iter=5, tol=None) )
        ])
        pipeline.fit()
        gs = GridSearchCV(pipeline, self.params, n_jobs=-1, cv=k_folds)

if __name__ == '__main__':
    main()