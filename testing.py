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
import pandas, numpy, textblob, string
from functools import reduce
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

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
    

with open('data/twitter_gender_data.json') as data:
        

     
        
        df=pandas.read_json(data)
        df.dropna(axis=0)
        df.set_index('id', inplace=True)
        df.head()
       
        features= [c for c in df.columns.values if c   in ['description','name']]
        
        target='gender'
        

        X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.1, random_state=42)
        X_train.head()
        tweet = Pipeline([
                ('selector', TextSelector(key='description')),
                
                ('words', CountVectorizer(analyzer='word'))
            ])
        name= Pipeline([
                ('selector', TextSelector(key='name')),
               
                ('words', CountVectorizer(analyzer='word'))
            ])

        tweet.fit_transform(X_train)
        name.fit_transform(X_train)
        feats = FeatureUnion([('tweet', tweet), 
                      
                      ('name', name)])

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
        
        hyperparameters = { 'classifier__C': [.0001,.001,.01,.1],
                    'classifier__gamma': [.01, .1 ,1]
                 
                  }
        clf = GridSearchCV(pipeline, hyperparameters, cv=5)
        clf.fit(X_train, y_train)
        print(clf.best_params_)
        
        clf.refit
        preds = clf.predict(X_test)
        

       
        print(preds)
        print( np.mean(preds == y_test))
        print(classification_report(y_test, preds))
        
      

       
        
        
        
    