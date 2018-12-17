try:
    import json
except ImportError:
    import simplejson as json
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
from sklearn.feature_extraction.text import TfidfTransformer
import re
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.utils.fixes import signature
from sklearn.naive_bayes import MultinomialNB
def normaliseData(x):
   
    scale=x.max(axis=0)
    return (x/scale, scale)  

def plotAccuracy(favouritesAcc,listed_acc,description_acc,tweet_acc,name_acc,screen_name,nameDescAcc,nameTweetAcc,nameScreen,nameCreatedAt,tweetDesc,tweetNameDesc,boop,doop):
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
    y = (nameScreen,nameCreatedAt,tweetDesc,tweetNameDesc,boop,doop)

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
    scores=clf.cv_results_['mean_test_score'].reshape(-1, 2).T
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
    if(name=="hashtag num "):
        X  = X.reshape(-1,1)
    else:
        (X, _) = normaliseData(X.reshape(-1,1))
   
    outerCV=KFold(n_splits=4, shuffle=True, random_state=21)
    hyperparameters={
            "C": [ .1, .5],
            
            
            "intercept_scaling": [.1, .5]#,
        
        
            #"tol": [1e-4, 1e-5, 1e-6]


            }
    clist=hyperparameters["C"]
    interlist=hyperparameters["intercept_scaling"]
    pipeline = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB()),
   ])
    cv = RepeatedKFold(n_splits=2, n_repeats=2)
    clf=GridSearchCV(estimator=pipeline, param_grid=hyperparameters, cv=cv)
    clf.fit(X, y)
  
   
    accuracy=cross_val_score(clf, X.reshape(-1,1), y, cv=cv).mean()
    recall=clf.score
    print(recall)
    recall=cross_val_score(clf, X.reshape(-1,1), y, cv=cv,scoring='recall').mean()
    precision=cross_val_score(clf, X.reshape(-1,1), y, cv=cv,scoring='precision').mean()
    predictions = cross_val_predict(clf, X, y, cv=outerCV)
   
    plotHeatMap(name, clf, clist, interlist)
    plotPrecisionRecall(predictions, y, name)
  
    f=open("scores.txt", "a+")
    f.write("scores for "+name)
    #f.write("accuracy: "+str(np.mean(accuracy))+" recall: "+str(np.mean(recall))+" precision: "+str(np.mean(precision))+"\n")
    f.close()

  
    #print(classification_report(y, predictions))
    return accuracy

   
def textClassification(X, y, name):
    # create a dataframe using texts and lables
    print("Model "+name)
    
  
    outerCV = KFold(n_splits=10, shuffle=True, random_state=21)
    
   
    hyperparameters={
    "clf__alpha": [ .1, 1],
    'tfidf__norm': ('l1', 'l2')#, 
    #'tfidf__use_idf': (True, False),  
    #'tfidf__sublinear_tf': (True, False)
    
  


    }
    alpha=hyperparameters["clf__alpha"]
    tfidf__norm=hyperparameters["tfidf__norm"]
    pipeline = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB()),
   ])
    print(pipeline.get_params().keys())
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
   
    plotHeatMap(name, clf, alpha, tfidf__norm)
    plotPrecisionRecall(predictions, y, name)
    return accuracy

def main():
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
            tweet_acc = textClassification(tweet_arr, gender_arr, "Tweet")
    

if __name__ == '__main__':
   main()



