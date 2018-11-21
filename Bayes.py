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
    with open('data/twitter_gender_data.json') as data:
       
        data = json.load(data)
        description = [d["description"] for d in data]
        name = [d["name"] for d in data]
        tweet = [d["tweet"] for d in data]
        print(tweet[1])
        gender = np.where(np.array([d["gender"] for d in data]) == 'M', 0, 1)
        Ttrain, Ttest, Gtrain, Gtest = train_test_split(description, gender, test_size=0.3, random_state=42)
        print(len(Ttrain))
        bayes(Ttrain, Ttest, Gtrain, Gtest)

if __name__ == '__main__':
    main()

