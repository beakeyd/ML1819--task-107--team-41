import nltk
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from langid.langid import LanguageIdentifier, model
import re

try:
    import json
except ImportError:
    import simplejson as json

with open('data/twitter_tweets_pruned.json') as data:
        identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        nltk.download('stopwords')
        nltk.download('punkt')
        tokenizer = RegexpTokenizer(r'\w+')
        data = json.load(data)
        copy = {}
        stop_words = set(stopwords.words('english'))
        i = 0
        porter = PorterStemmer()

        # stop word removal
        for key, values in data.items():
            copy[key] = []
            for val in values:
                tweet = tokenizer.tokenize(val)
                filtered_tweet = [] 
                for w in tweet: 
                    if w not in stop_words:
                        w = porter.stem(w)
                        filtered_tweet.append(w)
                tmp = " ".join(str(x) for x in filtered_tweet)
                tmp2=tmp.encode('ascii', 'ignore').decode("utf-8")
                lang=(identifier.classify(tmp2))[0]
                if lang=="en":
                    copy[key].append(tmp)
            i = i + 1
            print(i)

        with open('data/twitter_tweets_no_unicode_eng.json', 'w') as output1:
            json.dump(copy, output1)


#This function strips tweets only
def removeUnicodeAndLangId(data):
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    i=0
    j=0
    copy={}
    for d in data:
        listTweet=[]
        for tweet in data[d]:
            tweet=tweet.encode('ascii', 'ignore').decode("utf-8")
            lang=(identifier.classify(tweet))[0]
            if lang=="en":
                listTweet.append(tweet)

        if not len(listTweet) == 0:
            copy[d]=listTweet