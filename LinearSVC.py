#!/usr/bin/python
'''
The Purpose of this file is to apply the linear SVC 
technique on all forms of data (numberical+text)
'''



try:
    import json
except ImportError:
    import simplejson as json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from mpl_toolkits.mplot3d import Axes3D
import time

from datetime import datetime as dt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import  LinearSVC
from sklearn.model_selection import KFold
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas, numpy, textblob, string
from functools import reduce
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
#The multiple feature text classification code is based off https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines#
class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]
    
class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]

def main():
    
    with open('data/original_dataset.json') as data:
       
        data = json.load(data)

        # slice data
        created_at = np.array([d["created_at"] for d in data])
        favourites_count = np.array([d["favourites_count"]for d in data])
        listed_count = np.array([d["listed_count"] for d in data])
        description = np.array([d["description"] for d in data])
        tweet = np.array([d["tweet"] for d in data])
        name = np.array([d["name"] for d in data])
        screen_name = np.array([d["screen_name"] for d in data])
        gender = np.array([d["gender"] for d in data])

       

        #create models, plot and then get accuracy of models
        #created_at_acc = simpleFeature(created_at, gender)
        favourites_acc = simpleFeature(favourites_count, gender)
        
        #listed_acc = simpleFeature(listed_count, gender)   
        #description_acc = textClassification(description, gender)
        #tweet_acc = textClassification(tweet, gender)
        #name_acc = textClassification(name, gender)

        #plotAccuracy(created_at_acc, favourites_acc,
         #            listed_acc, description_acc,
          #           tweet_acc, name_acc, 'Accuracy')
    
    with open('data/original_dataset.json') as data:
        
        df=pandas.read_json(data)
        df.dropna(axis=0)
        df.set_index('id', inplace=True)
        df.head()
        #combinedFeatures("name", "description", df)
        #combinedFeatures("name", "tweet", df)
        #combinedFeatures("name", "screen_name", df)
        #combinedFeatures("name", "created_at", df)
        #combinedFeatures("tweet", "description", df)
        #combinedThreeTextFeatures("tweet", "name", "description", df)



def normaliseData(x):
   
    scale=x.max(axis=0)
    return (x/scale, scale)

def plotAccuracy(created_at_acc, favourites_acc,
                 listed_acc, description_acc,
                 tweet_acc, name_acc, graph_name):
    
    y = (created_at_acc, favourites_acc,
          listed_acc, description_acc,
         tweet_acc, name_acc)

    X_axis = ['created_at', 'favourites',
               'listed', 'description',
              'tweet', 'name']

    y_pos = np.arange(len(X_axis))

    plt.bar(y_pos, y, align='center', alpha=0.5)
    plt.xticks(y_pos, X_axis)
    plt.ylabel('Accuracy')
    plt.title('Accuracy of features')
    graph = 'plots/' + graph_name + '.png'
    #plt.show()

def plotSingleFeatureData(X, actY, predY, graph_name, xLabel):
    fig, ax = plt.subplots(figsize=(6,2))
    ax.scatter(X, actY, label='Data', marker='+')
    ax.scatter(X, predY, label='Prediction', marker='x')

    ax.set_xlabel('test', fontsize=12)
    ax.set_ylabel('Gender')
    ax.set_title(graph_name)
    graph = 'plots/' + graph_name + '1.png'
    fig.savefig(graph)


def created_at_model(created_at, y):
    # create Model
    (X, Xscale) = normaliseData(np.array(created_at).reshape(-1,1))
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1, random_state=42)
    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(Xtrain, ytrain)
   # getCAndGamma(clf, Xtrain, ytrain, 'created_at_model')
    # make predicitions
    predY = clf.predict(Xtest.reshape(-1, 1))
    #plot data, get and return accuracy of model
    print('created_at Model metrics: ')
    print(classification_report(ytest, predY))
    plotSingleFeatureData(Xtest, ytest, predY, 'Created_At', 'Posix Time Account Created At - Scaled between 0-1')
    accuracy = accuracy_score(ytest, predY)
    print(str(accuracy))
   # predY=printBestCGamma(clf, Xtrain, ytrain, Xtest, ytest, "single")
    accuracy = accuracy_score(ytest, predY)
    
    return accuracy



    
def simpleFeature(X, y):
    (X, _) = normaliseData(X.reshape(-1,1))
    kf=KFold(n_splits=10, shuffle=True, random_state=42)
    clf = LinearSVC(random_state=42, tol=1e-6, max_iter=1000)
    clf=hyperParameterTuning(clf, kf)

    accuracy=cross_val_score(clf, X.reshape(-1,1), y, cv=kf,n_jobs=-1)
    print(clf.best_params_)
    print (cross_val_score(clf, X.reshape(-1,1), y, cv=kf,n_jobs=-1,scoring='recall'))
    print (cross_val_score(clf, X.reshape(-1,1), y, cv=kf,n_jobs=-1,scoring='precision'))
    predictions = cross_val_predict(clf, X, y, cv=kf)
    
    print(classification_report(y, predictions))
    return accuracy


def textClassification(X, y):
    # create a dataframe using texts and lables
    

    kf=KFold(n_splits=10, shuffle=True, random_state=42)
    mean=0
    i=0
    for trainIndex, testIndex in kf.split(X):
        Xtrain, Xtest=X[trainIndex], X[testIndex]
        
        ytrain, ytest=y[trainIndex], y[testIndex]
        vectorizer = CountVectorizer(stop_words='english', max_df=0.2)
        Xtrain = vectorizer.fit_transform(Xtrain)
        Xtest = vectorizer.transform(Xtest)
        clf = LinearSVC(random_state=42, tol=1e-6, max_iter=1000)
        clf.fit(Xtrain, ytrain)
        predY=clf.predict(Xtest)
        accuracy=accuracy_score(ytest, predY)
        mean+=accuracy
        
        
        #plt.show()
    mean=mean/10
    print(mean)
    


    
    
    
  
    
    return Xtest, ytest, predY




#The multiple feature text classification code is based off https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines#
def combinedFeatures(x1, x2,df):
    
    print(x1+" and "+x2)
    graphName=(str(x1+" and "+x2))    
    features= [c for c in df.columns.values if c   in [x1,x2]]
    target='gender'
    

    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.1, random_state=42)
    X_train.head()
    if(isinstance(df.iloc[0][x1], str)):
        feat1 = Pipeline([
                ('selector', TextSelector(key=x1)),
                
                ('words', CountVectorizer(analyzer='word'))
            ])
    else:    
        feat1= Pipeline([
                ('selector', NumberSelector(key=x1)),
                
                ('words', StandardScaler())
            ])

    if(isinstance(df.iloc[0][x2], str)):
        feat2 = Pipeline([
                ('selector', TextSelector(key=x2)),
                
                ('words', CountVectorizer(analyzer='word'))
            ])
    else:    
        feat2= Pipeline([
                ('selector', NumberSelector(key=x2)),
                
                ('words', StandardScaler())
            ])


    feat1.fit_transform(X_train)
    feat2.fit_transform(X_train)
    feats = FeatureUnion([('feat1', feat1), 
                    
                    ('feat2', feat2)])
    feature_processing = Pipeline([('feats', feats)])
    feature_processing.fit_transform(X_train)
    pipeline = Pipeline([
        ('features',feats),
        ('classifier', LinearSVC(random_state=0, tol=1e-5)),
    ])
  
    # Fit and tune model
    pipeline.fit(X_train, y_train)
    preds=pipeline.predict(X_test)
    
    mean=np.mean(preds==y_test)
    print(mean)
    print(classification_report(y_test, preds))
    
    #printBestCGamma(pipeline, X_train, y_train, X_test, y_test, "multiple")
        
def hyperParameterTuning(model, kv):
    hyperparameters={
        "C": [0.01, .1, 1, 10],
        "dual": [True,False],
        "fit_intercept": [True, False],
        "intercept_scaling": [.1, 1, 10],
        "loss": ["hinge", "squared_hinge"],
        "max_iter": [100, 1000, 10000],
        "multi_class": ["ovr", "crammer_singer"],
      
        "tol": [1e-4, 1e-5, 1e-6, 1e-7]


    }
    clf=GridSearchCV(model, hyperparameters, cv=kv )
    
    return clf

#The multiple feature text classification code is based off https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines#
def combinedThreeTextFeatures(x1, x2, x3,df):
    print(x1+" and "+x2+" and "+x3)
    graphName=(str(x1+" and "+x2))
    
    features= [c for c in df.columns.values if c   in [x1,x2]]
    
    target='gender'
    

    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.1, random_state=42)
    X_train.head()
    tweet = Pipeline([
            ('selector', TextSelector(key=x1)),
            
            ('words', CountVectorizer(analyzer='word'))
        ])
    name= Pipeline([
            ('selector', TextSelector(key=x2)),
            
            ('words', CountVectorizer(analyzer='word'))
        ])
    description= Pipeline([
            ('selector', TextSelector(key=x2)),
            
            ('words', CountVectorizer(analyzer='word'))
        ])

    tweet.fit_transform(X_train)
    name.fit_transform(X_train)
    description.fit_transform(X_train)
    feats = FeatureUnion([('tweet', tweet), 
                    
                    ('name', name),
                    ('description', description)])

    feature_processing = Pipeline([('feats', feats)])
    feature_processing.fit_transform(X_train)
    pipeline = Pipeline([
        ('features',feats),
        ('classifier', LinearSVC(random_state=0, tol=1e-5)),
    ])
    
    # Fit and tune model
    pipeline.fit(X_train, y_train)
    preds=pipeline.predict(X_test)
    
    mean=np.mean(preds==y_test)
    print(mean)
    print(classification_report(y_test, preds))
    
    #printBestCGamma(pipeline, X_train, y_train, X_test, y_test, "multiple")
if __name__ == '__main__':
    main()
