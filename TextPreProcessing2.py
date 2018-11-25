import nltk
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import re

try:
    import json
except ImportError:
    import simplejson as json
jsonFile="data/DavidsPruned.json"
with open(jsonFile) as data:
        nltk.download('stopwords')
        nltk.download('punkt')
        tokenizer = RegexpTokenizer(r'\w+')
        data = json.load(data)
        copy = []
        stop_words = set(stopwords.words('english'))
        i = 0
        porter = PorterStemmer()

        # stop word removal
        for i in range(0, len(data) - 1):
            d = data[i]
            word_tokens_tweet = tokenizer.tokenize(d['tweet'])
            word_tokens_name = tokenizer.tokenize(d['name'])
            word_tokens_description = tokenizer.tokenize(d['description'])
            word_tokens_screen_name = tokenizer.tokenize(d['screen_name'])
        
            filtered_sentence_tweet = [w for w in word_tokens_tweet if not w in stop_words] 
            filtered_sentence_tweet = [] 
            for w in word_tokens_tweet: 
                if w not in stop_words:
                    w = porter.stem(w)
                    filtered_sentence_tweet.append(w)
            
            filtered_sentence_name = [w for w in word_tokens_name if not w in stop_words] 
            filtered_sentence_name = [] 
            for w in word_tokens_name: 
                if w not in stop_words: 
                    w = porter.stem(w)
                    filtered_sentence_name.append(w)

            filtered_sentence_description = [w for w in word_tokens_description if not w in stop_words] 
            filtered_sentence_description = [] 
            for w in word_tokens_description: 
                if w not in stop_words: 
                    w = porter.stem(w)
                    filtered_sentence_description.append(w)

            filtered_sentence_screen_name = [w for w in word_tokens_screen_name if not w in stop_words] 
            filtered_sentence_screen_name = [] 
            for w in word_tokens_screen_name: 
                if w not in stop_words:
                    w = porter.stem(w)
                    filtered_sentence_screen_name.append(w)
            
            d['tweet'] = " ".join(str(x) for x in filtered_sentence_tweet)
            d['name'] = " ".join(str(x) for x in filtered_sentence_name)
            d['description'] = " ".join(str(x) for x in filtered_sentence_description)
            d['screen_name'] = " ".join(str(x) for x in filtered_sentence_screen_name)
            copy.append(d)
            i = i + 1

            print(i)

        with open('data/new_twitter_gender_data.json', 'w') as output1:
                json.dump(copy, output1)
