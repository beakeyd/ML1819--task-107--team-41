#!/usr/bin/python
'''
The Purpose of this file is to apply the linear SVC 
technique on all forms of data (numberical+text)
'''



try:
    import json
except ImportError:
    import simplejson as json
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from mpl_toolkits.mplot3d import Axes3D
import time, mglearn
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
    acc =0
    with open('data/twitter_tweets_no_unicode.json') as data:
        with open('data/gender.json') as gender_data:
            
            data = json.load(data)
            gender_data = json.load(gender_data)
            gender_arr = []
            tweet_arr = []
        
            for key, value in data.items():
            
                for val in value:
                    tweet_arr.append(val)
                    if gender_data[key] == 'M':
                      
                        gender_arr.append(1)
                    else:
                        gender_arr.append(0)
                
            gender_arr=np.array(gender_arr)
            tweet_arr=np.array(tweet_arr)
            male, fem=0, 0
            for d in gender_arr:
                if d==1:
                    male+=1
                else:
                    fem+=1
           # acc = textClassification(tweet_arr, gender_arr, "Tweet LinearSVC")
    with open('data/original_dataset_nounicode.json') as totalDataset,open('data/original_dataset_nounicode.json') as totalDataset1,open("data/numb_Hashtag.json") as numbHashtag,open("data/twitter_Hashtag.json") as hashTagTweet, open("data/gender.json") as genderData:
        
        totalDataset = json.load(totalDataset)
        df=pandas.read_json(totalDataset1)
        # slice data
        #created_at = np.array([d["created_at"] for d in totalDataset])
        favourites_count = np.array([d["favourites_count"]for d in totalDataset])
        listed_count = np.array([d["listed_count"] for d in totalDataset])
        description = np.array([d["description"] for d in totalDataset])
        tweet = np.array([d["tweet"] for d in totalDataset])
        name = np.array([d["name"] for d in totalDataset])
        screen_name = np.array([d["screen_name"] for d in totalDataset])
        gender = np.array([d["gender"] for d in totalDataset])
      
       

        #create models, plot and then get accuracy of models
        #created_at_acc = simpleFeature(created_at, gender, "Created At")
        favouritesResults = simpleFeature(favourites_count, gender, "Favourites Count")
        
        listed_acc = simpleFeature(listed_count, gender, "Listed Count")   
        description_acc = textClassification(description, gender, "Description")
        tweet_acc = textClassification(tweet, gender, "Tweet")
        name_acc = textClassification(name, gender, "name")
        screen_name = textClassification(name, gender, "Screen Name")

        
        
        df.dropna(axis=0)
        df.set_index('id', inplace=True)
        df.head()
        nameDescAcc=combinedFeatures("name", "description", df)
        nameTweetAcc=combinedFeatures("name", "tweet", df)
        nameScreen=combinedFeatures("name", "screen_name", df)
        nameCreatedAt=combinedFeatures("name", "created_at", df)
        tweetDesc=combinedFeatures("tweet", "description", df)
        tweetNameDesc=combinedThreeTextFeatures("tweet", "name", "description", df)
        numbHashtag=json.load(numbHashtag)
        genderData=json.load(genderData)
        hashtagTweet=json.load(hashTagTweet)
        hashtagNumbAcc=hashtagNum(numbHashtag, genderData, "hashtag num ")
        hashtagTextAcc=hashtagText(hashtagTweet, genderData, "hashtag text ")
        
        plotAccuracy(favouritesResults, listed_acc, description_acc, tweet_acc, name_acc, screen_name, nameDescAcc, nameTweetAcc, nameScreen, nameCreatedAt, tweetDesc, tweetNameDesc, hashtagNumbAcc, hashtagTextAcc)
        

def normaliseData(x):
   
    scale=x.max(axis=0)
    return (x/scale, scale)

def plotAccuracy(favouritesAcc,listed_acc,description_acc,tweet_acc,name_acc,screen_name,nameDescAcc,nameTweetAcc,nameScreen,nameCreatedAt,tweetDesc,tweetNameDesc,hashtagNumb,hashTagText):
    plt.figure(figsize=(16, 16))
    plt.subplot(2,1,1)
  
    y = (favouritesAcc,listed_acc,description_acc,tweet_acc,name_acc,screen_name,nameDescAcc,nameTweetAcc)

    X_axis = ['favourites', 'listed count', 'description', 'tweet', 'name', 'screen name', 'name+desc', 'name+tweet']

    y_pos = np.arange(len(X_axis))

    plt.bar(y_pos, y, align='center', alpha=0.5)
    plt.xticks(y_pos, X_axis)
    plt.ylabel('Accuracy')
    plt.title('Accuracy of features')

    plt.subplot(2,1,2)
    y = (nameScreen,nameCreatedAt,tweetDesc,tweetNameDesc,hashtagNumb,hashTagText)

    X_axis = ['name+screen', 'name+createdat', 'tweet+desc', 'tweet+name+desc', 'number hashtags', 'hashtag text']

    y_pos = np.arange(len(X_axis))

    plt.bar(y_pos, y, align='center', alpha=0.5)
    plt.xticks(y_pos, X_axis)
    plt.ylabel('Accuracy')
    plt.title('Accuracy of features')
    graph = 'plots/Accuracy.png'
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

def plotSingleFeatureData(X, actY, predY, graph_name, xLabel, clf, clist, interlist):
    print(len(actY))
    fig, ax = plt.subplots(figsize=(6,2))
    ax.scatter(X, actY, label='Data', marker='+')
    ax.scatter(X, predY, label='Prediction', marker='x')

    ax.set_xlabel('test', fontsize=12)
    ax.set_ylabel('Gender')
    ax.set_title(graph_name)
    
    graph = 'plots/' + graph_name + '1.png'
    fig.savefig(graph)
   

#this function is no longer used
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
    #plotSingleFeatureData(Xtest, ytest, predY, 'Created_At', 'Posix Time Account Created At - Scaled between 0-1', clist, interlist)
    accuracy = accuracy_score(ytest, predY)
    print(str(accuracy))
   # predY=printBestCGamma(clf, Xtrain, ytrain, Xtest, ytest, "single")
    accuracy = accuracy_score(ytest, predY)
    
    return accuracy

#nested vs non nested sklearn tutorial    
def hashtagNum( data, data2,name):
    X, y=[],[]
    for values in data:
       
        if values in data2:
            if(data2[values]=="M"):
                y.append(1)
            else:
                y.append(0)
            X.append(data[values])
       
    X=np.array(X)
    
    y=np.array(y)
  
    accuracy=simpleFeature(X, y, name)
   
    return accuracy
def hashtagText( data, data2,name):
    X, y=[],[]
    for values in data:
        
        if values in data2:
            if(data2[values]=="M"):
                y.append(1)
            else:
                y.append(0)
            result=""
            for e in data[values]:
                result+=str(e)+" "
            
            X.append(result)
          
    X=np.array(X)
    y=np.array(y)
   
    textClassification(X, y, name)
    print("Model is " + name)
    return 0

#nested vs non nested sklearn tutorial    
def simpleFeature(X, y, name):
    print("Model is " + name)
    if(name=="hashtag num "):
        X  = X.reshape(-1,1)
    else:
        (X, _) = normaliseData(X.reshape(-1,1))
   
    outerCV=KFold(n_splits=4, shuffle=True, random_state=21)
    hyperparameters={
            "C": [ .1, .5, 1, 10],
            
            
            "intercept_scaling": [.1, .5, 1,10]
        
        
            


            }
    clist=hyperparameters["C"]
    interlist=hyperparameters["intercept_scaling"]
    model = LinearSVC(random_state=42 )
    cv = RepeatedKFold(n_splits=2, n_repeats=4)
    clf=GridSearchCV(estimator=model, param_grid=hyperparameters, cv=cv)
    clf.fit(X, y)
  
   
    accuracy=cross_val_score(clf, X.reshape(-1,1), y, cv=cv).mean()
    recall=clf.score
    print(recall)
    recall=cross_val_score(clf, X.reshape(-1,1), y, cv=cv,scoring='recall').mean()
    precision=cross_val_score(clf, X.reshape(-1,1), y, cv=cv,scoring='precision').mean()
    predictions = cross_val_predict(clf, X, y, cv=outerCV)
   
    plotHeatMap(name, clf, clist, interlist)
    plotPrecisionRecall(predictions, y, name)
  
    f=open("scoresLinearSVC.txt", "a+")
    f.write("scores for "+name)
    f.write("accuracy: "+str(accuracy)+" recall: "+str(recall)+" precision: "+str(precision)+"\n")
    f.close()

  
    #print(classification_report(y, predictions))
    return accuracy


def textClassification(X, y, name):
    # create a dataframe using texts and lables
    print("Model "+name)
    
  
    outerCV = KFold(n_splits=4, shuffle=True, random_state=21)
    vectorizer = CountVectorizer(stop_words='english', max_df=0.2)
    
    X = vectorizer.fit_transform(X)
    model = LinearSVC(random_state=42, tol=1e-6, max_iter=1000)
    hyperparameters={
            "C": [ .1, .5, 1, 10],
            
            
            "intercept_scaling": [.1, .5, 1,10]
        
        
            


            }
    clist=hyperparameters["C"]
    interlist=hyperparameters["intercept_scaling"]
    cv = RepeatedKFold(n_splits=2, n_repeats=4)
    clf=GridSearchCV(estimator=model, param_grid=hyperparameters, cv=cv)
    clf.fit(X, y)
    
    accuracy = cross_val_score(clf, X=X, y=y, cv=cv ).mean()
    print(accuracy)
    recall = cross_val_score(clf, X=X, y=y, cv=cv, scoring="recall").mean()
    print(recall)
    precision =cross_val_score(clf, X=X, y=y, cv=cv, scoring="precision").mean()
    print(precision)
    predictions = cross_val_predict(clf, X, y, cv=outerCV)
   
    plotHeatMap(name, clf, clist, interlist)
    plotPrecisionRecall(predictions, y, name)
    f=open("scoresLinearSVC.txt", "a+")
    f.write("scores for "+name)
    f.write("accuracy: "+str(accuracy)+" recall: "+str(recall)+" precision: "+str(precision)+"\n")
    f.close()
    return accuracy




#The multiple feature text classification code is based off https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines#
def combinedFeatures(x1, x2,df):
    

  
    outerCV = KFold(n_splits=4, shuffle=True, random_state=21)
    name=x1+" and "+x2
    graphName=(str(x1+" and "+x2))    
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
        ('classifier', LinearSVC(random_state=0, tol=1e-5)),
    ])
    hyperparameters={
        "classifier__C": [ .1, .5, 1,  10]
        ,
        
        "classifier__intercept_scaling": [.1, .5, 1, 10]
       
        
      
       


        }
    clist=hyperparameters["classifier__C"]
    interlist=hyperparameters["classifier__intercept_scaling"]
    cv = RepeatedKFold(n_splits=2, n_repeats=4)
    clf=GridSearchCV(estimator=pipeline, param_grid=hyperparameters, cv=cv)
    clf.fit(X, y)
    
    accuracy = cross_val_score(clf, X=X, y=y, cv=cv ).mean()
    print(accuracy)
    recall = cross_val_score(clf, X=X, y=y, cv=cv, scoring="recall").mean()
    print(recall)
    precision =cross_val_score(clf, X=X, y=y, cv=cv, scoring="precision").mean()
    print(precision)
    predictions = cross_val_predict(clf, X, y, cv=outerCV)
   
    plotHeatMap(name, clf, clist, interlist)
    plotPrecisionRecall(predictions, y, name)
    f=open("scoresLinearSVC.txt", "a+")
    f.write("scores for "+graphName)
    f.write("accuracy: "+str(accuracy)+" recall: "+str(recall)+" precision: "+str(precision)+"\n")
    f.close()
    
    return accuracy
    
    
        


#The multiple feature text classification code is based off https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines#
def combinedThreeTextFeatures(x1, x2, x3,df):
    print(x1+" and "+x2+" and "+x3)
    graphName=(str(x1+" and "+x2+" and "+x3))
    outerCV = KFold(n_splits=4, shuffle=True, random_state=21)
    features= [c for c in df.columns.values if c   in [x1,x2]]
    
    target='gender'
    

    X, y = df[features], df[target]
    X.head()
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
    
    tweet.fit_transform(X)
    name.fit_transform(X)
    description.fit_transform(X)
   
    feats = FeatureUnion([('tweet', tweet), 
                    
                    ('name', name),
                    ('description', description)])

    feature_processing = Pipeline([('feats', feats)])
    feature_processing.fit_transform(X)
    pipeline = Pipeline([
        ('features',feats),
        ('classifier', LinearSVC(random_state=0, tol=1e-5)),
    ])
    hyperparameters={
        "classifier__C": [ .1, .5, 1,  10]
        ,
        
        "classifier__intercept_scaling": [.1, .5, 1, 10]
       
      


        }
    clist=hyperparameters["classifier__C"]
    interlist=hyperparameters["classifier__intercept_scaling"]
    cv = RepeatedKFold(n_splits=2, n_repeats=4)
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
    f=open("scoresLinearSVC.txt", "a+")
    f.write("scores for "+graphName)
    f.write("accuracy: "+str(accuracy)+" recall: "+str(recall)+" precision: "+str(precision)+"\n")
    f.close()
    return accuracy
if __name__ == '__main__':
    main()
