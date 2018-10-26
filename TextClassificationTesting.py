# This script will determine which is the best technique for text classification
# So we can know which features are actually the best prediction of gender.
# Base taken from https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
try:
    import json
except ImportError:
    import simplejson as json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split
import pandas, numpy, textblob, string


def main():
    print("wefwef")
    with open('data/twitter_gender_data.json') as data:
        data = json.load(data)

        description = [d["name"] for d in data]
        gender = [d["gender"] for d in data]

        # create a dataframe using texts and lables
        trainDF = pandas.DataFrame()
        trainDF['text'] = description
        trainDF['label'] = gender

        # split the dataset into training and validation datasets 
        train_x, test_x, train_y, test_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], test_size=0.1)

        vectorizer = CountVectorizer(stop_words='english', max_df=0.2)
        train_x = vectorizer.fit_transform(train_x)
        test_x = vectorizer.transform(test_x)

        model = svm.SVC(C=1.0,kernel='linear')
        model.fit(train_x, train_y)

        preds = model.predict(test_x)
        print(classification_report(test_y, preds))

if __name__ == '__main__':
    main()