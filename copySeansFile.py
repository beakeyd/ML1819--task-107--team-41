import nltk
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import re

try:
    import json
except ImportError:
    import simplejson as json
jsonFile="data/tweets_lang_pruned.json"
with open(jsonFile) as data:
        nltk.download('stopwords')
        nltk.download('punkt')
        tokenizer = RegexpTokenizer(r'\w+')
        data = json.load(data)
        copy = {}
        stop_words = set(stopwords.words('english'))
        i = 0
        porter = PorterStemmer()
        j=0
        # stop word removal
        for d in data:
            if i>2:
                break
            i+=1
            listTweet=[]
            for tweet in data[d]:  
                if j>2:
                    break
                j+=1
                word_tokens_tweet = tokenizer.tokenize(tweet)
                
            
                filtered_sentence_tweet = [w for w in word_tokens_tweet if not w in stop_words] 
                filtered_sentence_tweet = [] 
                for w in word_tokens_tweet: 
                    if w not in stop_words:
                        w = porter.stem(w)
                        filtered_sentence_tweet.append(w)
                
                
                
                tweet = " ".join(str(x) for x in filtered_sentence_tweet)
                listTweet.append(tweet)
            data[d]=listTweet
            copy[d]=listTweet
            j=0

        with open('data/tweets_only_pruned.json', 'w') as output1:
                json.dump(copy, output1)
