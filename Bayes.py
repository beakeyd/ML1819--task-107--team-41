from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from sklearn import metrics
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split

try:
   import json
except ImportError:
   import simplejson as json

def bayes(tweetTrain, tweetTest, genderTrain, genderTest):
   text_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB()),
   ])
   text_clf.fit(tweetTrain, genderTrain) 
   predicted = text_clf.predict(tweetTest)
   np.mean(predicted == genderTest)
   print(classification_report(genderTest, predicted))
   accuracy = accuracy_score(genderTest, predicted)
   print(str(accuracy))

def main():
    with open('data/twitter_tweets_pruned.json') as data:
        with open('data/gender.json') as gender_data:
            data = json.load(data)
            gender_data = json.load(gender_data)
            gender_arr = []
            text_arr = []
            for key, value in data.items():
                for val in value:
                    text_arr.append(val)
                    if gender_data[key] == 'M':
                        gender_arr.append(1)
                    else:
                        gender_arr.append(0)
            
            Ttrain, Ttest, Gtrain, Gtest = train_test_split(text_arr, gender_arr, test_size=0.33, random_state=42)
            bayes(Ttrain, Ttest, Gtrain, Gtest)

if __name__ == '__main__':
   main()



