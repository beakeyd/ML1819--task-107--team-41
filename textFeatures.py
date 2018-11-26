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
jsonFile="data/DavidsPruned.json"


#outdated function
'''
def removeNonEnglish(data):
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

    i=0
    listD=[]
    i=0
    j=0
    for d in data:
        
        tweets=[]
        for tweet in data[d]:  
                   
            lang=(identifier.classify(tweet))[0]
            
            if lang=="en":
                tweets.append(tweet)
        data[d]=tweets
            
           
            #print(len(listD))
    with open('data/tweets_lang_pruned.json', 'w+') as output:
        json.dump(data, output)
'''     


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
            
           
        if not len(listTweet) == 0 :   
            copy[d]=listTweet
        
    with open('data/twitter_tweets_no_unicode.json', 'w') as output1:
                json.dump(copy, output1)

#this function strips the main dataset 
def stripRest(data):
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

    i=0
 
    j=0
    copy=[]
    for d in data:
       
        
        tweet=d["tweet"]
        name=d['name']
        description=d['description']
        screen_name=d['screen_name'] 
        tweet=tweet.encode('ascii', 'ignore').decode("utf-8")
        name=name.encode('ascii', 'ignore').decode("utf-8")
        description=description.encode('ascii', 'ignore').decode("utf-8")
        tweet=tweet.encode('ascii', 'ignore').decode("utf-8")
        d['tweet']=tweet
        d['name']=name
        d['description']=description
        d['screen_name']=screen_name
        copy.append(d)
       
    
        
    with open('data/original_dataset_nounicode.json', 'w') as output1:
                json.dump(copy, output1)
       
             
    


def main():
  
   
    
    
    with open('data/twitter_tweets_pruned.json') as data:
        data=json.load(data)
        removeUnicodeAndLangId(data)
    #with open('data/original_dataset.json') as data:
    #    data=json.load(data)
    #    stripRest(data)










if __name__ == '__main__':
    main()
