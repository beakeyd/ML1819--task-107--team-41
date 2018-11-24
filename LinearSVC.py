#!/usr/bin/python
'''
The Purpose of this file is to apply the linear SVC 
technique on all forms of data (numberical+text)
'''



#!/usr/bin/python
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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import svm
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
    
    with open('data/Davids.json') as data:
       
      
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
        
        # slice data
    with open('data/Davids.json') as data:
        data = json.load(data)
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
       
        #testL = [[d["name"], d["description"]] for d in data]

        #create models, plot and then get accuracy of models
        created_at_acc = created_at_model(created_at, gender)
        favourites_acc = favourites_count_model(favourites_count, gender)
        listed_acc = listed_count_model(listed_count, gender)   
        description_acc = description_model(description, gender)
        tweet_acc = tweet_model(tweet, gender)
        name_acc = name_model(name, gender)

        plotAccuracy(created_at_acc, favourites_acc,
                     listed_acc, description_acc,
                     tweet_acc, name_acc, 'Accuracy')
    
   
      
        

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
    clf = svm.NuSVC(kernel='poly', nu=.99)
   
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

def favourites_count_model(favourites_count, y):
    # create Model
    (X, _) = normaliseData(np.array(favourites_count).reshape(-1,1))
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,test_size=0.1, random_state=42)
    clf = svm.NuSVC(kernel='poly', degree=3, nu=.1)
    #getCAndGamma(clf, X, y, 'favourites_count_model')
    clf.fit(Xtrain, ytrain)
    # make predicitions
    predY = clf.predict(Xtest.reshape(-1,1))
    #plot data, get and return accuracy of model
    print('favourites_count Model metrics: ')
    print(classification_report(ytest, predY))
    plotSingleFeatureData(Xtest, ytest, predY, 'Favourites_Count', 'Number of Favourites on User Tweets - Scaled between 0-1')

    accuracy = accuracy_score(ytest, predY)
    print(str(accuracy))
 #   predY=printBestCGamma(clf, Xtrain, ytrain, Xtest, ytest, "single")
    accuracy = accuracy_score(ytest, predY)
    return accuracy

def listed_count_model(listed_count, y):
    # create Model
    (X, _) = normaliseData(np.array(listed_count).reshape(-1,1))
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1, random_state=42)
    clf = svm.NuSVC(kernel='poly', degree=3,nu= .1)
   
    clf.fit(X, y)

    # make predicitions
    predY = clf.predict(Xtest.reshape(-1,1))
    #plot data, get and return accuracy of model
   
   
    print(classification_report(ytest, predY))
    #plotSingleFeatureData(Xtest, ytest, predY, 'Listed_Count', 'Number of Lists User appears on - Scaled between 0-1')
    
    accuracy = accuracy_score(ytest, predY)
    print(str(accuracy))
  #  predY=printBestCGamma(clf, Xtrain, ytrain, Xtest, ytest, "single")
    accuracy = accuracy_score(ytest, predY)
    return accuracy

''' 
    Models that ARE doing text classification
'''

def description_model(description, gender):
    print('Description Model Metrics: ')
    Xtest, ytest, predY = textClassification(description, gender, 0.5, 4, 'description_model')
    
   
    
    accuracy = accuracy_score(ytest, predY)
   
    return accuracy

def tweet_model(tweet, gender):
    print('Tweet Model Metrics: ')
    Xtest, ytest, predY = textClassification(tweet, gender, 0.1, 10, 'tweet_model')
    
    
    
    accuracy = accuracy_score(ytest, predY)

    return accuracy
    
def name_model(name, gender):
    print('Name Model Metrics: ')
    Xtest, ytest, predY = textClassification(name, gender, 0.35, 7, 'name_model')
   
    accuracy = accuracy_score(ytest, predY)
    
    return accuracy

def textClassification(X, y, gamma_val, C_val, name):
    # create a dataframe using texts and lables
    trainDF = pandas.DataFrame()
    trainDF['text'] = X
    trainDF['label'] = y

    # split the dataset into training and validation datasets 
    
    Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(trainDF['text'], trainDF['label'], test_size=0.1, random_state=42)
    vectorizer = CountVectorizer(stop_words='english', max_df=0.2)
    Xtrain = vectorizer.fit_transform(Xtrain)
    Xtest = vectorizer.transform(Xtest)
    
    #tf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')
    #Xtrain = tf.fit_transform(Xtrain)
       
    model = svm.SVC(kernel='poly', C=C_val, gamma=gamma_val)
#    getCAndGamma(model, X, y, name)
    model.fit(Xtrain, ytrain)
    predY = model.predict(Xtest)
    accuracy = accuracy_score(ytest, predY)
    print(accuracy)
    print(classification_report(ytest, predY))
    #if(name=="tweet_model"):
     #   getCAndGamma(model, Xtrain, ytrain, "test")

    
  
    
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
        ('classifier', svm.SVC(kernel='linear', C=1, gamma=1)),
    ])
    
    # Fit and tune model
    pipeline.fit(X_train, y_train)
    preds=pipeline.predict(X_test)
    
    mean=np.mean(preds==y_test)
    print(mean)
    print(classification_report(y_test, preds))
    
    #printBestCGamma(pipeline, X_train, y_train, X_test, y_test, "multiple")
        
    

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
        ('classifier', svm.SVC(kernel='linear', C=10, gamma=1)),
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
