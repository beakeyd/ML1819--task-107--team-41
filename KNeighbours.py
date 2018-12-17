#!/usr/bin/python
'''
The Purpose of this file is to apply the KNeighnours algorithm to numerical data 
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
import time, mglearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from datetime import datetime as dt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import  LinearSVC
from sklearn.model_selection import KFold
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, RepeatedKFold
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
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.utils.fixes import signature
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
    
    with open('data/original_pruned_With_hashtagNum.json') as totalDataset,open('data/original_pruned_With_hashtagNum.json') as totalDataset1 :
       
        totalDataset=json.load(totalDataset)
       
        
        df=pandas.read_json(totalDataset1)
        df.dropna(axis=0)
        df.set_index('id', inplace=True)
        df.head()
       
       
        # slice data
       
        favourites_count = np.array([d["favourites_count"]for d in totalDataset])
        listed_count = np.array([d["listed_count"] for d in totalDataset])
        hashtagNum= np.array([d["hashtagNumb"] for d in totalDataset])
        gender = np.array([d["gender"] for d in totalDataset])
      
       
        
        #create models, plot and then get accuracy of models
       
        favouritesResults = simpleFeature(favourites_count, gender, "Favourites Count KNeighbour")
        
        listed_acc = simpleFeature(listed_count, gender, "Listed Count KNeighbour")
       
        numberAcc=simpleFeature(hashtagNum, gender, "Hashtag Count KNeighbour" )
        favouritesListedAcc=combinedFeatures("favourites_count", "listed_count","Favourite and Listed Accuracy KNeighbour", df)
        favouritesNumberAcc=combinedFeatures("favourites_count", "hashtagNumb","Favourite and Hashtag Accuracy Kneighbour",df)
        listedNumberAcc=combinedFeatures("listed_count", "hashtagNumb","Listed Count and Hashtag Accuracy Kneighbour",df)
        
        allThreeAcc=combinedThreeFeatures("hashtagNumb", "listed_count", "favourites_count","Favourite Listed and Hashtag Numb KNeighbour",df)
       
        
        
       
    
       
        plotAccuracy(favouritesResults, listed_acc, numberAcc,  favouritesListedAcc, favouritesNumberAcc, listedNumberAcc,allThreeAcc)
    

def normaliseData(x):
   
    scale=x.max(axis=0)
    return (x/scale, scale)

def plotAccuracy(favouritesAcc,listed_acc,numberAcc,favouritesListedAcc,favouritesNumberAcc,listedNumberAcc,allThreeAcc):
    plt.figure(figsize=(16, 16))
    
  
    y = (favouritesAcc,listed_acc,numberAcc,favouritesListedAcc,favouritesNumberAcc,listedNumberAcc,allThreeAcc)

    X_axis = ['favouritesAcc', 'listed_acc', 'numberAcc', 'favouritesListedAcc', 'favouritesNumberAcc', 'listedNumberAcc ', 'allThreeAcc']

    y_pos = np.arange(len(X_axis))

    plt.bar(y_pos, y, align='center', alpha=0.5)
    plt.xticks(y_pos, X_axis)
    plt.ylabel('Accuracy Kneighbours')
    plt.title('Accuracy of features')

   
    graph = 'plots/Accuracy Kneighbours.png'
    plt.savefig(graph)
    plt.close()
    #plt.show()

def plotHeatMap(graphName, clf, clist, interlist):
    plt.figure(figsize=(8, 8))
    scores=clf.cv_results_['mean_test_score'].reshape(-1, 4).T
    heatmap=mglearn.tools.heatmap(scores, xlabel="C", ylabel="boop", cmap="viridis", fmt="%.3f", xticklabels=clist, yticklabels=interlist)
    plt.colorbar(heatmap)
    graph = 'plots/' + graphName+'HyperParam.png'
    plt.savefig(graph)
    plt.close()
    
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
def plotPrecisionRecall(predictions, y, graphName):
    precision, recall, _ = precision_recall_curve(predictions, y)
    average_precision = average_precision_score(predictions, y)
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                if 'step' in signature(plt.fill_between).parameters
                else {})
    plt.step(recall, precision, color='b', alpha=0.2,
            where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    graph = 'plots/' + graphName+'PrecisionRecall.png'
    plt.savefig(graph)
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
    plt.close()



#nested vs non nested sklearn tutorial    
def simpleFeature(X, y, name):
    print("Model is " + name)
    if(name=="Hashtag Numb KNeighbour"):
        X  = X.reshape(-1,1)
    else:
        (X, _) = normaliseData(X.reshape(-1,1))
   
    outerCV=KFold(n_splits=4, shuffle=True, random_state=21)
    hyperparameters={
        "n_neighbors": [ 1,5,10]
        ,
        
        "leaf_size": [10, 20, 30, 50],
       
        "p": [1,2]
      
      


        }
    neighbour=hyperparameters["n_neighbors"]
    leaflist=hyperparameters["leaf_size"]
    model = KNeighborsClassifier()
    cv = RepeatedKFold(n_splits=2, n_repeats=2)
    clf=GridSearchCV(estimator=model, param_grid=hyperparameters, cv=cv)
    clf.fit(X, y)
  
   
    accuracy=cross_val_score(clf, X.reshape(-1,1), y, cv=cv).mean()
    recall=clf.score
    print(recall)
    recall=cross_val_score(clf, X.reshape(-1,1), y, cv=cv,scoring='recall').mean()
    precision=cross_val_score(clf, X.reshape(-1,1), y, cv=cv,scoring='precision').mean()
    predictions = cross_val_predict(clf, X, y, cv=outerCV)
   
    plotHeatMap(name, clf, neighbour, leaflist)
    plotPrecisionRecall(predictions, y, name)
  
    f=open("scores.txt", "a+")
    f.write("scores for "+name)
    #f.write("accuracy: "+str(np.mean(accuracy))+" recall: "+str(np.mean(recall))+" precision: "+str(np.mean(precision))+"\n")
    f.close()

  
    #print(classification_report(y, predictions))
    return accuracy







#The multiple feature text classification code is based off https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines#
def combinedFeatures(x1, x2, graphName,df):
    outerCV = KFold(n_splits=10, shuffle=True, random_state=21)
   
    
    features= [c for c in df.columns.values if c   in [x1,x2]]
    target='gender'
    

    X, y = df[features], df[target]

    X.head()
    
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


    feat1.fit_transform(X)
    feat2.fit_transform(X)
    feats = FeatureUnion([('feat1', feat1), 
                    
                    ('feat2', feat2)])
    feature_processing = Pipeline([('feats', feats)])
    feature_processing.fit_transform(X)
    pipeline = Pipeline([
        ('features',feats),
        ('classifier', KNeighborsClassifier()
        ),
    ])
    hyperparameters={
        "classifier__n_neighbors": [ 1,5,10, 20]
        ,
        
        "classifier__leaf_size": [10, 20, 30, 50]#,
       
       # "classifier__p": [1,2]


        }
    neighbours=hyperparameters["classifier__n_neighbors"]
    leafList=hyperparameters["classifier__leaf_size"]
    cv = RepeatedKFold(n_splits=2, n_repeats=2)
    clf=GridSearchCV(estimator=pipeline, param_grid=hyperparameters, cv=cv)
    clf.fit(X, y)
    
    accuracy = cross_val_score(clf, X=X, y=y, cv=cv ).mean()
    print(accuracy)
    recall = cross_val_score(clf, X=X, y=y, cv=cv, scoring="recall").mean()
    print(recall)
    precision =cross_val_score(clf, X=X, y=y, cv=cv, scoring="precision").mean()
    print(precision)
    predictions = cross_val_predict(clf, X, y, cv=outerCV)
   
    plotHeatMap(graphName, clf, neighbours, leafList)
    plotPrecisionRecall(predictions, y, graphName)
    
    return accuracy


    
    
        


#The multiple feature  classification code is based off https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines#
def combinedThreeFeatures(x1, x2, x3,graphName, df):
    print(x1+" and "+x2+" and "+x3)
   
    outerCV = KFold(n_splits=10, shuffle=True, random_state=21)
    features= [c for c in df.columns.values if c   in [x1,x2, x3]]
    
    target='gender'
    

    X, y = df[features], df[target]
    X.head()
    feat1= Pipeline([
                ('selector', NumberSelector(key=x1)),
                
                ('words', StandardScaler())
            ])
    feat2= Pipeline([
                ('selector', NumberSelector(key=x2)),
                
                ('words', StandardScaler())
            ])
    feat3= Pipeline([
                ('selector', NumberSelector(key=x3)),
                
                ('words', StandardScaler())
            ])
    
    feat1.fit_transform(X)
    feat2.fit_transform(X)
    feat3.fit_transform(X)
   
    feats = FeatureUnion([('tweet', feat1), 
                    
                    ('name', feat2),
                    ('description', feat3)])

    feature_processing = Pipeline([('feats', feats)])
    feature_processing.fit_transform(X)
    pipeline = Pipeline([
        ('features',feats),
        ('classifier', LinearSVC(random_state=0, tol=1e-5)),
    ])
    hyperparameters={
        "classifier__C": [ .1, 1]
        ,
        
        "classifier__intercept_scaling": [.1, 1],
       
       # "classifier__max_iter": [100, 1000, 10000],
      
        #"classifier__tol": [1e-4, 1e-5, 1e-6]


        }
    clist=hyperparameters["classifier__C"]
    interlist=hyperparameters["classifier__intercept_scaling"]
    cv = RepeatedKFold(n_splits=2, n_repeats=2)
    clf=GridSearchCV(estimator=pipeline, param_grid=hyperparameters, cv=cv)
    clf.fit(X, y)
    
    accuracy = cross_val_score(clf, X=X, y=y, cv=cv ).mean()
    print(accuracy)
    recall = cross_val_score(clf, X=X, y=y, cv=cv, scoring="recall").mean()
    print(recall)
    precision =cross_val_score(clf, X=X, y=y, cv=cv, scoring="precision").mean()
    print(precision)
    predictions = cross_val_predict(clf, X, y, cv=outerCV)
   
    plotHeatMap(graphName, clf, clist, interlist)
    plotPrecisionRecall(predictions, y, graphName)
    return accuracy
    
    #printBestCGamma(pipeline, X_train, y_train, X_test, y_test, "multiple")
if __name__ == '__main__':
    main()
