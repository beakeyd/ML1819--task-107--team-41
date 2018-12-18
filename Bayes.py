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

def plotAccuracy(AllTweets,screenName,description,tweet_acc):
    plt.figure(figsize=(16, 16))
  
  
    y = (AllTweets,screenName,description,tweet_acc)

    X_axis = ['All Tweets ', 'Screen Name', 'Description', 'Tweet']

    y_pos = np.arange(len(X_axis))

    plt.bar(y_pos, y, align='center', alpha=0.5)
    plt.xticks(y_pos, X_axis)
    plt.ylabel('Accuracy')
    plt.title('Accuracy of features for Bayes')

   
    graph = 'plots/AccuracyBayes.png'
    plt.savefig(graph)
    plt.close()
    #plt.show()

def plotHeatMap(graphName, clf, clist, interlist):
    plt.figure(figsize=(8, 8))
    scores=clf.cv_results_['mean_test_score'].reshape(-1, 3).T
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

   
def textClassification(X, y, name):
    # create a dataframe using texts and lables
    print("Model "+name)
    
    outerCV = KFold(n_splits=2, shuffle=True, random_state=21)
    
    hyperparameters={
    "clf__alpha": [ .1,  1, 10],
    'tfidf__norm': ('l1', 'l2') 
    
    }
    alpha=hyperparameters["clf__alpha"]
    tfidf__norm=hyperparameters["tfidf__norm"]
    pipeline = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB()),
   ])
    
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
   
    #plotHeatMap(name, clf, alpha, tfidf__norm)
    plotPrecisionRecall(predictions, y, name)
    f=open("scoresBayes.txt", "a+")
    f.write("scores for "+name)
    f.write(" accuracy: "+str(accuracy)+" recall: "+str(recall)+" precision: "+str(precision)+"\n")
    f.close()
    return accuracy

def main():
   
    acc=0
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
            acc = textClassification(tweet_arr, gender_arr, "Tweet Bayes")
            
          
 
 
    
    with open('data/original_dataset_nounicode.json') as data:
        with open('data/gender.json') as gender_data:
            
            data = json.load(data)
            gender_data = json.load(gender_data)
            gender_arr = []
            screen_name_arr = []
            name_arr = []
            description_arr = []
            for elm in data:
                screen_name_arr.append(elm["screen_name"])
                name_arr.append(elm["name"])
                description_arr.append(elm["description"])
                _id = elm["id"]
                if gender_data[str(_id)] == 'M':
                    gender_arr.append(1)
                else:
                    gender_arr.append(0)
            gender_arr=np.array(gender_arr)
            name_arr=np.array(name_arr)
            screen_name_arr=np.array(screen_name_arr)
          
            description_arr=np.array(description_arr)
            male, fem=0, 0
            for d in gender_arr:
                if d==1:
                    male+=1
                else:
                    fem+=1
            screenNameAcc = textClassification(screen_name_arr, gender_arr, "screen_name_Bayes")
            genderNameAcc = textClassification(name_arr, gender_arr, "name_Bayes")
            descriptionAcc = textClassification(description_arr, gender_arr, "description_bayes")
    plotAccuracy(acc, screenNameAcc, genderNameAcc, descriptionAcc)
if __name__ == '__main__':
   main()



